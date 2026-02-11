#!/usr/bin/env python3
"""
Visit-aware matching for ADNI MRI scans.

Goal:
- Map each .nii scan in --scan-dir to its clinical visit (VISCODE) using ADSXLIST (RID + VISDATE),
  then join to DXSUM (RID + VISCODE) to fetch the diagnosis for that same visit.

Inputs:
- --scan-dir: directory containing .nii files named like: ADNI_123_S_4567_<anything>_YYYY-MM-DD*.nii
- --adsxlist-file: ADSXLIST_*.csv (must include: RID, VISCODE, VISCODE2, VISDATE)
- --dxsum-file: DXSUM_*.csv (must include: RID, VISCODE, VISCODE2, EXAMDATE, DIAGNOSIS)
- [optional] --blchange-file: BLCHANGE_*.csv to attach additional visit-level variables (RID, VISCODE, EXAMDATE, etc.)

Outputs (written next to --scan-dir):
- dataset_visitaware.csv: matched scans with RID, scan_date, VISCODE, EXAMDATE, DIAGNOSIS, etc.
- dataset_visitaware_unmatched.csv: scans that could not be matched to a visit/diagnosis (with reasons)

Matching policy:
1) Parse RID & scan_date from each filename.
2) Find the nearest VISDATE in ADSXLIST for the same RID within ±t days (default t=7), preferring exact same day.
3) Resolve to that row's VISCODE (fallback to VISCODE2 if VISCODE is missing).
4) Join with DXSUM on (RID, VISCODE). If multiple rows, pick the one with EXAMDATE closest to scan_date.
5) If no visit within tolerance, record as unmatched (no label is better than a wrong label).

Notes:
- You can raise --date-tolerance-days if scheduling gaps are larger at your site, but keep it small (≤14) to avoid drift.
- If ADSXLIST lacks some scans, consider adding other imaging listings as additional sources for (RID, VISCODE, date).

Usage example:
python visit_aware_match.py \
  --scan-dir /path/to/scans \
  --adsxlist-file /path/to/ADSXLIST_05Apr2025.csv \
  --dxsum-file /path/to/DXSUM_05Apr2025.csv \
  --blchange-file /path/to/BLCHANGE_05Apr2025.csv \
  --date-tolerance-days 7
"""

import sys
import argparse
from pathlib import Path
import re
from datetime import datetime
import pandas as pd


FNAME_PATTERN = re.compile(
    r'^ADNI_(?P<ptid>\d+_S_\d+)_.*?_(?P<date>\d{4}-\d{2}-\d{2}).*\.nii$', re.IGNORECASE
)


def parse_filename(fname: str):
    """
    Extract (RID:int, scan_date:Timestamp, PTID:str) from filename.
    Expected: ADNI_<PTID>_*_<YYYY-MM-DD>*.nii   where PTID=123_S_4567 and RID=4567
    """
    m = FNAME_PATTERN.match(fname)
    if not m:
        raise ValueError("unrecognized filename format")
    ptid = m.group("ptid")                   # e.g., 123_S_4567
    rid = int(ptid.split("_")[-1])           # 4567
    scan_date = pd.to_datetime(m.group("date"), format="%Y-%m-%d", errors="raise")
    return rid, scan_date, ptid


def load_adsxlist(path: str):
    df = pd.read_csv(path)
    # minimal columns
    required = {"RID", "VISCODE", "VISCODE2", "VISDATE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ADSXLIST file missing columns: {missing}")
    out = df.loc[:, ["RID", "VISCODE", "VISCODE2", "VISDATE"]].copy()
    out["RID"] = out["RID"].astype(int)
    out["VISDATE"] = pd.to_datetime(out["VISDATE"], errors="coerce")
    out = out.dropna(subset=["VISDATE"])
    # prefer VISCODE, but keep VISCODE2 as backup
    return out


def load_dxsum(path: str):
    df = pd.read_csv(path)
    required = {"RID", "VISCODE", "VISCODE2", "EXAMDATE", "DIAGNOSIS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DXSUM file missing columns: {missing}")
    out = df.loc[:, ["RID", "VISCODE", "VISCODE2", "EXAMDATE", "DIAGNOSIS"]].copy()
    out["RID"] = out["RID"].astype(int)
    out["EXAMDATE"] = pd.to_datetime(out["EXAMDATE"], errors="coerce")
    out = out.dropna(subset=["EXAMDATE"])
    return out


def load_blchange(path: str):
    df = pd.read_csv(path)
    # keep a conservative set to avoid column collisions
    keep = [c for c in ["RID", "VISCODE", "VISCODE2", "EXAMDATE"] if c in df.columns]
    others = [c for c in df.columns if c not in keep]
    # prefix other columns to avoid name collisions on merge
    df_pref = df[keep + others].copy()
    for c in others:
        if c not in {"RID", "VISCODE", "VISCODE2", "EXAMDATE"}:
            df_pref.rename(columns={c: f"BL_{c}"}, inplace=True)
    if "RID" in df_pref.columns:
        df_pref["RID"] = df_pref["RID"].astype(int)
    if "EXAMDATE" in df_pref.columns:
        df_pref["EXAMDATE"] = pd.to_datetime(df_pref["EXAMDATE"], errors="coerce")
    return df_pref


def build_scan_table(scan_dir: str):
    paths = list(Path(scan_dir).glob("*.nii"))
    rows = []
    for p in paths:
        try:
            rid, sdate, ptid = parse_filename(p.name)
            rows.append({"filepath": str(p.resolve()), "RID": rid, "PTID": ptid, "scan_date": sdate})
        except Exception as e:
            rows.append({"filepath": str(p.resolve()), "RID": None, "PTID": None, "scan_date": None, "parse_error": str(e)})
    df = pd.DataFrame(rows)
    return df


def pick_viscode(viscode: str, viscode2: str):
    # prefer VISCODE; fallback to VISCODE2 if VISCODE is null/empty
    if pd.notna(viscode) and str(viscode).strip() != "":
        return str(viscode).strip()
    if pd.notna(viscode2) and str(viscode2).strip() != "":
        return str(viscode2).strip()
    return None


def match_scans_to_visits(df_scans: pd.DataFrame, df_ads: pd.DataFrame, tolerance_days: int):
    """
    For each scan, find the ADSXLIST row (same RID) with VISDATE closest to scan_date within ±tolerance_days.
    Prefer exact day (diff=0). Returns df with chosen VISCODE and VISDATE; unmatched are returned separately.
    """
    scans = df_scans.dropna(subset=["RID", "scan_date"]).copy()
    ads = df_ads.copy()

    # Cross-join per RID via merge on RID, then compute |date diff|
    merged = scans.merge(ads, on="RID", how="left", suffixes=("", "_ads"))
    merged["days_diff"] = (merged["VISDATE"] - merged["scan_date"]).abs().dt.days

    # keep only within tolerance
    within = merged[merged["days_diff"] <= tolerance_days].copy()

    if within.empty:
        return pd.DataFrame(columns=list(scans.columns) + ["VISCODE_RESOLVED", "VISDATE"]), df_scans

    # Prefer exact match (0 days), otherwise the minimum days_diff
    # Sort so that exact matches come first, then smallest diff, then stable order by VISDATE
    within["is_exact"] = within["days_diff"].eq(0).astype(int)
    within["VISCODE_RESOLVED"] = within.apply(
        lambda r: pick_viscode(r.get("VISCODE"), r.get("VISCODE2")), axis=1
    )
    # Drop rows where we still don't have a visit code
    within = within[within["VISCODE_RESOLVED"].notna()].copy()

    if within.empty:
        return pd.DataFrame(columns=list(scans.columns) + ["VISCODE_RESOLVED", "VISDATE"]), df_scans

    within.sort_values(
        by=["filepath", "is_exact", "days_diff", "VISDATE"],
        ascending=[True, False, True, True],
        inplace=True,
        kind="mergesort",
    )

    # Take the best row per scan (filepath)
    best = within.groupby("filepath", as_index=False).first()

    # Merge best back to original scans
    chosen = scans.merge(
        best[["filepath", "VISCODE_RESOLVED", "VISDATE"]],
        on="filepath",
        how="left",
    )

    # Determine unmatched: those with no VISCODE_RESOLVED
    unmatched_mask = chosen["VISCODE_RESOLVED"].isna()
    unmatched = df_scans[df_scans["filepath"].isin(chosen.loc[unmatched_mask, "filepath"])].copy()
    unmatched["reason"] = f"No visit within ±{tolerance_days} days or missing VISCODE"

    matched = chosen[~unmatched_mask].copy()
    return matched, unmatched


def attach_diagnosis(df_matched: pd.DataFrame, df_dx: pd.DataFrame):
    """
    Join matched scans (RID + VISCODE_RESOLVED) to DXSUM to get DIAGNOSIS/EXAMDATE.
    If multiple DXSUM rows for the same visit, select the EXAMDATE closest to scan_date.
    """
    # Prepare DXSUM with a resolved visit code
    dx = df_dx.copy()
    dx["VISCODE_RESOLVED"] = dx.apply(lambda r: pick_viscode(r.get("VISCODE"), r.get("VISCODE2")), axis=1)
    dx = dx.dropna(subset=["VISCODE_RESOLVED"])

    # Candidate join
    cand = df_matched.merge(
        dx[["RID", "VISCODE_RESOLVED", "EXAMDATE", "DIAGNOSIS"]],
        on=["RID", "VISCODE_RESOLVED"],
        how="left",
        suffixes=("", "_dx"),
    )

    if cand.empty:
        return pd.DataFrame(columns=list(df_matched.columns) + ["EXAMDATE", "DIAGNOSIS"]), df_matched.assign(reason="No DXSUM rows for (RID,VISCODE)")

    # Pick the DXSUM row with EXAMDATE closest to scan_date
    cand["dx_days_diff"] = (cand["EXAMDATE"] - cand["scan_date"]).abs().dt.days
    cand.sort_values(
        by=["filepath", "dx_days_diff", "EXAMDATE"],
        ascending=[True, True, True],
        inplace=True,
        kind="mergesort",
    )
    best = cand.groupby("filepath", as_index=False).first()

    # Unmatched if DIAGNOSIS is null
    unmatched = best[best["DIAGNOSIS"].isna()].copy()
    unmatched["reason"] = "No diagnosis found in DXSUM for (RID,VISCODE)"

    matched = best[best["DIAGNOSIS"].notna()].copy()
    return matched, unmatched[["filepath", "reason"]]


def maybe_attach_blchange(df_in: pd.DataFrame, df_bl: pd.DataFrame):
    """
    Optional: attach BLCHANGE variables on (RID, VISCODE). If duplicates, pick EXAMDATE closest to scan_date.
    """
    if df_bl is None or df_bl.empty:
        return df_in

    bl = df_bl.copy()
    bl["VISCODE_RESOLVED"] = bl.apply(lambda r: pick_viscode(r.get("VISCODE"), r.get("VISCODE2")), axis=1)
    bl = bl.dropna(subset=["VISCODE_RESOLVED"])

    cand = df_in.merge(
        bl.drop(columns=["VISCODE", "VISCODE2"], errors="ignore"),
        left_on=["RID", "VISCODE_RESOLVED"],
        right_on=["RID", "VISCODE_RESOLVED"],
        how="left",
        suffixes=("", "_bl"),
    )

    # If multiple BLCHANGE rows per visit, pick closest EXAMDATE to scan_date
    if "EXAMDATE_bl" in cand.columns:
        cand["bl_days_diff"] = (cand["EXAMDATE_bl"] - cand["scan_date"]).abs().dt.days
        cand.sort_values(
            by=["filepath", "bl_days_diff", "EXAMDATE_bl"],
            ascending=[True, True, True],
            inplace=True,
            kind="mergesort",
        )
        cand = cand.groupby("filepath", as_index=False).first()

    return cand


def main():
    ap = argparse.ArgumentParser(description="Visit-aware matching for ADNI MRI scans using VISCODE.")
    ap.add_argument("--scan-dir", required=True)
    ap.add_argument("--adsxlist-file", required=True)
    ap.add_argument("--dxsum-file", required=True)
    ap.add_argument("--blchange-file", default=None)
    ap.add_argument("--date-tolerance-days", type=int, default=7,
                    help="Max |VISDATE - scan_date| to accept for visit mapping (default: 7)")
    args = ap.parse_args(sys.argv[1:])

    # Load data
    df_scans = build_scan_table(args.scan_dir)
    df_ads = load_adsxlist(args.adsxlist_file)
    df_dx = load_dxsum(args.dxsum_file)
    df_bl = load_blchange(args.blchange_file) if args.blchange_file else None

    # Match scans to visits
    matched_vis, unmatched_vis = match_scans_to_visits(df_scans, df_ads, args.date_tolerance_days)

    # Attach diagnosis
    matched_dx, unmatched_dx = attach_diagnosis(matched_vis, df_dx)

    # Optional: attach BLCHANGE
    final = maybe_attach_blchange(matched_dx, df_bl)

    # Select and order columns for output
    base_cols = ["filepath", "RID", "PTID", "scan_date", "VISCODE_RESOLVED", "VISDATE", "EXAMDATE", "DIAGNOSIS"]
    keep_cols = base_cols + [c for c in final.columns if c not in base_cols]
    final = final.loc[:, [c for c in keep_cols if c in final.columns]].sort_values(["RID", "scan_date"])

    # Combine unmatched
    unmatched = pd.concat([unmatched_vis[["filepath", "reason"]], unmatched_dx[["filepath", "reason"]]], axis=0, ignore_index=True)
    unmatched = unmatched.dropna(subset=["filepath"]).drop_duplicates()

    out_dir = Path(args.scan_dir)
    out_csv = out_dir / "dataset_visitaware.csv"
    un_csv = out_dir / "dataset_visitaware_unmatched.csv"

    if not final.empty:
        final.to_csv(out_csv, index=False)
        print(f"Wrote {len(final)} matched rows -> {out_csv}")
    else:
        print("No matched rows to write.")

    if not unmatched.empty:
        unmatched.to_csv(un_csv, index=False)
        print(f"Wrote {len(unmatched)} unmatched rows -> {un_csv}")
    else:
        print("No unmatched rows.")


if __name__ == "__main__":
    main()
