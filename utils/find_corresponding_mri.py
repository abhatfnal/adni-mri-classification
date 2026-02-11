#!/usr/bin/env python3
import pandas as pd
import sys, os, re, json
from datetime import datetime

# ---------- PET helpers ----------
PTID_REGEX = re.compile(r'(\d{3}_S_\d{3,5})')  # allow 3–5 digits after _S_

# ADNI placeholder dates that should be ignored if seen in filenames
INVALID_YMD = {"19770703", "19000101"}
INVALID_YMD_DASH = {"1977-07-03", "1900-01-01"}

def parse_ptid_from_path(path: str):
    s = path.replace(os.sep, '/')
    m = PTID_REGEX.search(s)
    return m.group(1) if m else None

def parse_pet_date_from_json(pet_nii: str):
    stem, _ = os.path.splitext(pet_nii)
    jpath = stem + ".json"
    if not os.path.isfile(jpath):
        return None
    try:
        with open(jpath, "r") as f:
            meta = json.load(f)
        # Prefer AcquisitionDateTime if present
        for key in ("AcquisitionDateTime", "AcquisitionDate", "SeriesDate", "StudyDate"):
            val = meta.get(key)
            if not val:
                continue
            s = str(val)
            m = re.search(r'(\d{4}-\d{2}-\d{2})', s)
            if m and m.group(1) not in INVALID_YMD_DASH:
                return datetime.strptime(m.group(1), "%Y-%m-%d")
            m = re.search(r'(\d{8})', s)
            if m and m.group(1) not in INVALID_YMD:
                return datetime.strptime(m.group(1), "%Y%m%d")
    except Exception:
        pass
    return None

def parse_date_from_string(path: str):
    s = path.replace(os.sep, '/')
    m = re.search(r'(\d{4}-\d{2}-\d{2})', s)
    if m and m.group(1) not in INVALID_YMD_DASH:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass
    m = re.search(r'(\d{8})', s)
    if m and m.group(1) not in INVALID_YMD:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            pass
    return None

def ptid_rid(ptid):
    """Return the integer RID from a PTID like 033_S_0514 or 052_S_10168."""
    if not ptid:
        return None
    try:
        return int(ptid.split('_')[-1])  # works for 3–5 digits
    except Exception:
        return None

# ---------- MRI CSV ----------
def load_mri_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    fp_col  = next((cols[c] for c in ("filepath","file_path","path") if c in cols), None)
    rid_col = next((cols[c] for c in ("rid","subject_rid","subjectid","subject_id") if c in cols), None)
    sdate   = next((cols[c] for c in ("scan_date","mri_date","date") if c in cols), None)
    edate   = next((cols[c] for c in ("exam_date","visit_date") if c in cols), None)
    if not fp_col or not rid_col or (not sdate and not edate):
        sys.exit(0)

    keep = [fp_col, rid_col] + ([sdate] if sdate else []) + ([edate] if edate else [])
    df = df[keep].copy()
    df.columns = ["filepath","rid"] + (["scan_date"] if sdate else []) + (["exam_date"] if edate else [])

    # Parse date from scan_date else exam_date
    base_date_col = "scan_date" if "scan_date" in df.columns else "exam_date"
    df["date"] = pd.to_datetime(df[base_date_col], errors="coerce")

    # Clean RID
    df["rid"] = pd.to_numeric(df["rid"], errors="coerce")
    df = df.dropna(subset=["rid"])
    df["rid"] = df["rid"].astype(int)

    # PTID from filepath (now supports 3–5 digit RIDs)
    df["filepath"] = df["filepath"].astype(str)
    df["ptid"] = df["filepath"].apply(parse_ptid_from_path)

    # Quality flags
    def _is_good_t1(p):
        s = os.path.basename(p).lower()
        good = any(k in s for k in [
            "mprage","mp-rage","ir-fspgr","fspgr","spgr","bravo",
            "sagittal_mprage","accelerated_sagittal_mprage","sag_ir-fspgr"
        ])
        bad  = any(k in s for k in ["scout","aahead","localizer","loc_","tof","fieldmap","calib"])
        return good and not bad
    df["good_t1"] = df["filepath"].apply(_is_good_t1)

    def _has_y(p):
        d, b = os.path.dirname(p), os.path.basename(p)
        stem = os.path.splitext(b)[0]
        return os.path.isfile(os.path.join(d, f"y_{stem}.nii")) or os.path.isfile(os.path.join(d, f"iy_{stem}.nii"))
    df["has_y"] = df["filepath"].apply(_has_y)

    # Drop rows with no valid date (we can still sort by has_y/good_t1/date if present)
    df = df.dropna(subset=["date"])
    return df[["filepath","rid","date","ptid","good_t1","has_y"]]

# ---------- Matching ----------
def choose_best(cand: pd.DataFrame, pet_date: datetime, tol_days):
    if cand.empty:
        return None
    c = cand.copy()

    if pet_date is not None:
        c["diff"] = (c["date"] - pet_date).abs().dt.days
        if tol_days is not None:
            c = c[c["diff"] <= tol_days]
            if c.empty:
                return None
        # prefer: has_y > good_t1 > closest date > most recent
        c = c.sort_values(by=["has_y","good_t1","diff","date"],
                          ascending=[False, False, True, False])
    else:
        # No PET date available → pick best-quality, most recent MRI
        c = c.sort_values(by=["has_y","good_t1","date"],
                          ascending=[False, False, False])

    return None if c.empty else str(c.iloc[0]["filepath"])

def find_mri_for_pet(pet_nii: str, mri_df: pd.DataFrame):
    ptid = parse_ptid_from_path(pet_nii)
    pet_date = parse_pet_date_from_json(pet_nii) or parse_date_from_string(pet_nii)

    # Build candidate sets in strict → loose order
    candidates = []

    # 1) Exact PTID match (most reliable)
    if ptid:
        candidates.append(mri_df[mri_df["ptid"] == ptid])

    # 2) Exact numeric RID match (handles 3–5 digit RIDs; ADNI1/GO/2/3/4)
    rid = ptid_rid(ptid)
    if rid is not None:
        candidates.append(mri_df[mri_df["rid"] == rid])

    # 3) (Optional) same RID AND same site code inferred from PTID
    if ptid and rid is not None:
        site = ptid.split('_')[0]
        # constrain by both site string and RID to avoid partial/substring mistakes
        site_mask = mri_df["filepath"].str.contains(fr"/{site}_S_", regex=True, na=False)
        rid_mask  = (mri_df["rid"] == rid)
        candidates.append(mri_df[site_mask & rid_mask])

    # Try progressively larger tolerances, then no tolerance (if PET date exists)
    for cand in candidates:
        for tol in (180, 365, None):
            out = choose_best(cand, pet_date, tol)
            if out:
                return out
    return None

# ---------- Main ----------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(0)
    pet_nii_path, mri_csv_path = sys.argv[1], sys.argv[2]
    try:
        df = load_mri_csv(mri_csv_path)
        path = find_mri_for_pet(pet_nii_path, df)
        if path:
            print(path)
    except Exception:
        sys.exit(0)
