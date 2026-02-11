#!/usr/bin/env python3
"""
Hybrid visit-aware matching for ADNI MRI scans.

Priority:
1) Visit-aware: map scan -> VISCODE using ADSXLIST (RID + VISDATE within ±visit_tolerance)
   then join to DXSUM on (RID, VISCODE) for DIAGNOSIS.
2) Fallback (optional): if no VISCODE found, match scan date directly to nearest EXAMDATE
   in DXSUM for same RID within ±fallback_tolerance.

Outputs:
- dataset_visitaware.csv          (matched rows with match_mode = ['visit'|'date_fallback'])
- dataset_visitaware_unmatched.csv (remaining unmatched with reasons and parse errors)
- Prints a concise summary including unmatched-reason counts.

Usage:
python visit_aware_match_hybrid.py \
  --scan-dir /path/to/scans \
  --adsxlist-file /path/to/ADSXLIST_05Apr2025.csv \
  --dxsum-file /path/to/DXSUM_05Apr2025.csv \
  --blchange-file /path/to/BLCHANGE_05Apr2025.csv \
  --visit-tolerance-days 14 \
  --fallback-tolerance-days 60
"""



import sys
import argparse
from pathlib import Path
import re
import collections
import pandas as pd

# Accept .nii or .nii.gz
FNAME_PATTERN = re.compile(
    r'^ADNI_(?P<ptid>\d+_S_\d+)_.*?_(?P<date>\d{4}-\d{2}-\d{2}).*\.nii(?:\.gz)?$', re.IGNORECASE
)

def parse_filename(fname: str):
    """
    Extract (RID:int, scan_date:Timestamp, PTID:str) from filename:
    ADNI_<PTID>_*_<YYYY-MM-DD>*.nii[.gz]
    """
    m = FNAME_PATTERN.match(fname)
    if not m:
        raise ValueError("unrecognized filename format")
    ptid = m.group("ptid")                   # e.g., 123_S_4567
    rid = int(ptid.split("_")[-1])           # 4567
    scan_date = pd.to_datetime(m.group("date"), format="%Y-%m-%d", errors="raise")
    return rid, scan_date, ptid

def build_scan_table(scan_dir: str):
    paths = list(Path(scan_dir).rglob("*.nii")) + list(Path(scan_dir).rglob("*.nii.gz"))
    rows = []
    for p in paths:
        try:
            rid, sdate, ptid = parse_filename(p.name)
            rows.append({"filepath": str(p.resolve()), "RID": rid, "PTID": ptid, "scan_date": pd.Timestamp(sdate)})
        except Exception as e:
            rows.append({"filepath": str(p.resolve()), "RID": None, "PTID": None, "scan_date": None, "parse_error": str(e)})
    return pd.DataFrame(rows)

def pick_viscode(viscode: str, viscode2: str):
    if pd.notna(viscode) and str(viscode).strip():
        return str(viscode).strip()
    if pd.notna(viscode2) and str(viscode2).strip():
        return str(viscode2).strip()
    return None

def load_adsxlist(path: str):
    df = pd.read_csv(path)
    required = {"RID", "VISCODE", "VISCODE2", "VISDATE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ADSXLIST file missing columns: {missing}")
    out = df.loc[:, ["RID", "VISCODE", "VISCODE2", "VISDATE"]].copy()
    out["RID"] = out["RID"].astype(int)
    out["VISDATE"] = pd.to_datetime(out["VISDATE"], errors="coerce")
    out = out.dropna(subset=["VISDATE"])
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
    # resolve visit code
    out["VISCODE_RESOLVED"] = out.apply(lambda r: pick_viscode(r.get("VISCODE"), r.get("VISCODE2")), axis=1)
    return out

def load_blchange(path: str):
    df = pd.read_csv(path)
    keep = [c for c in ["RID", "VISCODE", "VISCODE2", "EXAMDATE"] if c in df.columns]
    others = [c for c in df.columns if c not in keep]
    df_pref = df[keep + others].copy()
    for c in others:
        if c not in {"RID", "VISCODE", "VISCODE2", "EXAMDATE"}:
            df_pref.rename(columns={c: f"BL_{c}"}, inplace=True)
    if "RID" in df_pref.columns:
        df_pref["RID"] = df_pref["RID"].astype(int)
    if "EXAMDATE" in df_pref.columns:
        df_pref["EXAMDATE"] = pd.to_datetime(df_pref["EXAMDATE"], errors="coerce")
    df_pref["VISCODE_RESOLVED"] = df_pref.apply(lambda r: pick_viscode(r.get("VISCODE"), r.get("VISCODE2")), axis=1)
    return df_pref

def match_scans_to_visits(df_scans, df_ads, tolerance_days):
    scans = df_scans.dropna(subset=["RID", "scan_date"]).copy()
    ads = df_ads.copy()

    merged = scans.merge(ads, on="RID", how="left", suffixes=("", "_ads"))
    merged["days_diff"] = (merged["VISDATE"] - merged["scan_date"]).abs().dt.days

    within = merged[merged["days_diff"] <= tolerance_days].copy()
    if within.empty:
        chosen = scans.copy()
        chosen["VISCODE_RESOLVED"] = pd.NA
        chosen["VISDATE"] = pd.NaT
        unmatched = df_scans.copy()
        unmatched["reason"] = f"No ADSXLIST VISDATE within ±{tolerance_days} days"
        return chosen, unmatched

    within["is_exact"] = within["days_diff"].eq(0).astype(int)
    within["VISCODE_RESOLVED"] = within.apply(lambda r: pick_viscode(r.get("VISCODE"), r.get("VISCODE2")), axis=1)
    within = within[within["VISCODE_RESOLVED"].notna()].copy()

    if within.empty:
        chosen = scans.copy()
        chosen["VISCODE_RESOLVED"] = pd.NA
        chosen["VISDATE"] = pd.NaT
        unmatched = df_scans.copy()
        unmatched["reason"] = "ADSXLIST rows found but VISCODE missing"
        return chosen, unmatched

    within.sort_values(
        by=["filepath", "is_exact", "days_diff", "VISDATE"],
        ascending=[True, False, True, True],
        inplace=True,
        kind="mergesort",
    )
    best = within.groupby("filepath", as_index=False).first()

    chosen = scans.merge(best[["filepath", "VISCODE_RESOLVED", "VISDATE"]], on="filepath", how="left")
    return chosen, pd.DataFrame(columns=["filepath", "reason"])

def attach_diagnosis_visit(chosen, df_dx):
    # Join using (RID, VISCODE_RESOLVED); pick EXAMDATE closest to scan_date
    dx = df_dx.dropna(subset=["VISCODE_RESOLVED"]).copy()
    cand = chosen.merge(
        dx[["RID", "VISCODE_RESOLVED", "EXAMDATE", "DIAGNOSIS"]],
        on=["RID", "VISCODE_RESOLVED"],
        how="left",
        suffixes=("", "_dx"),
    )
    if cand.empty:
        return pd.DataFrame(), chosen.assign(reason="No DXSUM rows for (RID,VISCODE)")

    cand["dx_days_diff"] = (cand["EXAMDATE"] - cand["scan_date"]).abs().dt.days
    cand.sort_values(
        by=["filepath", "dx_days_diff", "EXAMDATE"],
        ascending=[True, True, True],
        inplace=True,
        kind="mergesort",
    )
    best = cand.groupby("filepath", as_index=False).first()

    matched = best[best["DIAGNOSIS"].notna()].copy()
    matched["match_mode"] = "visit"
    unmatched = best[best["DIAGNOSIS"].isna()][["filepath"]].copy()
    unmatched["reason"] = "No diagnosis found in DXSUM for (RID,VISCODE)"
    return matched, unmatched

def attach_diagnosis_date_fallback(df_scans, df_dx, fallback_days):
    """
    Date-only fallback: for scans with no VISCODE match,
    pick DXSUM row with EXAMDATE closest to scan_date within ±fallback_days.
    """
    scans = df_scans.dropna(subset=["RID", "scan_date"]).copy()
    dx = df_dx.copy()

    cand = scans.merge(dx[["RID", "EXAMDATE", "DIAGNOSIS"]], on="RID", how="left")
    cand["days_diff"] = (cand["EXAMDATE"] - cand["scan_date"]).abs().dt.days
    cand = cand[cand["days_diff"] <= fallback_days].copy()

    if cand.empty:
        return pd.DataFrame(), scans.assign(reason=f"No EXAMDATE within ±{fallback_days} days (DXSUM)")

    cand.sort_values(
        by=["filepath", "days_diff", "EXAMDATE"],
        ascending=[True, True, True],
        inplace=True,
        kind="mergesort",
    )
    best = cand.groupby("filepath", as_index=False).first()
    matched = best[best["DIAGNOSIS"].notna()].copy()
    matched["match_mode"] = "date_fallback"
    unmatched = best[best["DIAGNOSIS"].isna()][["filepath"]].copy()
    unmatched["reason"] = f"DXSUM within ±{fallback_days} days but DIAGNOSIS missing"
    return matched, unmatched

def maybe_attach_blchange(df_in, df_bl):
    if df_bl is None or df_bl.empty:
        return df_in
    bl = df_bl.dropna(subset=["VISCODE_RESOLVED"]).copy()
    cand = df_in.merge(
        bl.drop(columns=["VISCODE", "VISCODE2"], errors="ignore"),
        on=["RID", "VISCODE_RESOLVED"],
        how="left",
        suffixes=("", "_bl"),
    )
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

def summarize_unmatched(df_unmatched):
    if df_unmatched is None or df_unmatched.empty:
        print("No unmatched rows.")
        return
    counts = collections.Counter(df_unmatched["reason"].fillna("unknown"))
    print("Unmatched breakdown:")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")

def main():
    ap = argparse.ArgumentParser(description="Hybrid visit-aware matching with date fallback.")
    ap.add_argument("--scan-dir", required=True)
    ap.add_argument("--adsxlist-file", required=True)
    ap.add_argument("--dxsum-file", required=True)
    ap.add_argument("--blchange-file", default=None)
    ap.add_argument("--visit-tolerance-days", type=int, default=7,
                    help="Max |VISDATE - scan_date| for visit mapping (default: 7)")
    ap.add_argument("--fallback-tolerance-days", type=int, default=60,
                    help="Max |EXAMDATE - scan_date| for date-only fallback (default: 60)")
    args = ap.parse_args(sys.argv[1:])

    # Load inputs
    df_scans = build_scan_table(args.scan_dir)
    df_ads = load_adsxlist(args.adsxlist_file)
    df_dx = load_dxsum(args.dxsum_file)
    df_bl = load_blchange(args.blchange_file) if args.blchange_file else None

    # Separate parse errors early
    parse_errs = df_scans[df_scans["parse_error"].notna()] if "parse_error" in df_scans.columns else pd.DataFrame()
    df_scans_valid = df_scans[df_scans.get("parse_error").isna()] if "parse_error" in df_scans.columns else df_scans

    # Visit-aware mapping
    chosen, unmatched_visit_map = match_scans_to_visits(df_scans_valid, df_ads, args.visit_tolerance_days)

    # Visit-aware diagnosis
    matched_visit, unmatched_visit_dx = attach_diagnosis_visit(chosen[chosen["VISCODE_RESOLVED"].notna()], df_dx)

    # Fallback for those without VISCODE
    need_fallback = chosen[chosen["VISCODE_RESOLVED"].isna()]
    matched_fallback, unmatched_fallback = attach_diagnosis_date_fallback(need_fallback, df_dx, args.fallback_tolerance_days)

    # Combine matched
    matched_all = pd.concat([matched_visit, matched_fallback], ignore_index=True, sort=False)

    # Attach BLCHANGE if available (only meaningful for visit-aware rows)
    if df_bl is not None and not df_bl.empty:
        matched_all = maybe_attach_blchange(matched_all, df_bl)

    # Order columns
    base_cols = ["filepath", "RID", "PTID", "scan_date", "VISCODE_RESOLVED", "VISDATE", "EXAMDATE", "DIAGNOSIS", "match_mode"]
    keep_cols = base_cols + [c for c in matched_all.columns if c not in base_cols]
    matched_all = matched_all.loc[:, [c for c in keep_cols if c in matched_all.columns]].sort_values(["RID", "scan_date"])

    # Collect unmatched
    unmatched_parts = []
    if not parse_errs.empty:
        tmp = parse_errs[["filepath", "parse_error"]].copy()
        tmp.rename(columns={"parse_error": "reason"}, inplace=True)
        unmatched_parts.append(tmp)
    if unmatched_visit_map is not None and not unmatched_visit_map.empty:
        unmatched_parts.append(unmatched_visit_map[["filepath", "reason"]])
    if unmatched_visit_dx is not None and not unmatched_visit_dx.empty:
        unmatched_parts.append(unmatched_visit_dx[["filepath", "reason"]])
    if unmatched_fallback is not None and not unmatched_fallback.empty:
        unmatched_parts.append(unmatched_fallback[["filepath", "reason"]])

    unmatched_all = pd.concat(unmatched_parts, ignore_index=True).drop_duplicates() if unmatched_parts else pd.DataFrame(columns=["filepath","reason"])

    out_dir = Path(args.scan_dir)
    out_csv = out_dir / "dataset_visitaware.csv"
    un_csv = out_dir / "dataset_visitaware_unmatched.csv"

    if not matched_all.empty:
        matched_all.to_csv(out_csv, index=False)
        print(f"Wrote {len(matched_all)} matched rows -> {out_csv}")
    else:
        print("No matched rows to write.")

    if not unmatched_all.empty:
        unmatched_all.to_csv(un_csv, index=False)
        print(f"Wrote {len(unmatched_all)} unmatched rows -> {un_csv}")
        summarize_unmatched(unmatched_all)
    else:
        print("No unmatched rows.")

if __name__ == "__main__":
    main()
