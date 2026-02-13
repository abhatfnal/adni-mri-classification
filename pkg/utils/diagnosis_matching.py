import pandas as pd

from tqdm import tqdm


def match_diagnosis(df_scan, df_diagnostic, tolerance):
        """
        Adds diagnosis column to scan dataframe by matching it with temporally closest visit
        in the diagnositc dataframe. 
        """
        
        to_concat = []
    
        df_scan = df_scan.copy()
        df_diagnostic = df_diagnostic.copy()

        # Ensure datetime dtype once
        df_diagnostic["EXAMDATE"] = pd.to_datetime(df_diagnostic["EXAMDATE"])
        df_scan["image_date"] = pd.to_datetime(df_scan["image_date"])

        for subj in tqdm(df_scan["subject_id"].unique()):

            # Get subject visits and scans
            subj_vis = df_diagnostic[df_diagnostic["PTID"] == subj].copy()
            subj_scans = df_scan[df_scan["subject_id"] == subj].copy()

            # Drop NaN values
            subj_vis = subj_vis.dropna(subset=["EXAMDATE"])
            subj_scans = subj_scans.dropna(subset=["image_date"])
            
            # Merge_asof requires sorting by the key
            subj_vis = subj_vis.sort_values("EXAMDATE")
            subj_scans = subj_scans.sort_values("image_date")

            # Match nearest withing tolerance
            nearest = pd.merge_asof(
                subj_scans,
                subj_vis[["EXAMDATE", "DIAGNOSIS"]],
                left_on="image_date",
                right_on="EXAMDATE",
                direction="nearest",
                tolerance=pd.Timedelta(days=tolerance),
            )

            # Get matched and append to list
            matched = nearest[nearest["DIAGNOSIS"].notna()].copy()
            matched = matched.rename(columns={"DIAGNOSIS": "diagnosis", "EXAMDATE":"exam_date"})
            matched = matched[["image_id", "image_date", "subject_id", "exam_date", "group", "modality", "diagnosis"]]  
    
            to_concat.append(matched)

            # Matched ids
            matched_ids = set(matched["image_id"])

            # If all scans matched, continue
            if len(matched_ids) == len(subj_scans):
                continue

            # Remaining scans (no visit within tolerance)
            remaining = subj_scans[~subj_scans["image_id"].isin(matched_ids)].copy()

            # Get closest visits before
            before = pd.merge_asof(
                remaining,
                subj_vis[["EXAMDATE", "DIAGNOSIS"]],
                left_on="image_date",
                right_on="EXAMDATE",
                direction="backward",   # visit on/before scan
                allow_exact_matches=True,
            ).rename(columns={"EXAMDATE": "EXAMDATE_before", "DIAGNOSIS": "DIAGNOSIS_before"})

            # Get closest visits after
            after = pd.merge_asof(
                remaining,
                subj_vis[["EXAMDATE", "DIAGNOSIS"]],
                left_on="image_date",
                right_on="EXAMDATE",
                direction="forward",    # visit on/after scan
                allow_exact_matches=True,
            ).rename(columns={"EXAMDATE": "EXAMDATE_after", "DIAGNOSIS": "DIAGNOSIS_after"})

            
            # Bracket: closest visit before + closest visit after!
            bracket = before.merge(
                after[["image_id", "EXAMDATE_after", "DIAGNOSIS_after"]],
                on="image_id",
                how="inner",
            )

            # need both sides, and diagnosis must agree
            bracket = bracket[
                bracket["DIAGNOSIS_before"].notna()
                & bracket["DIAGNOSIS_after"].notna()
                & (bracket["DIAGNOSIS_before"] == bracket["DIAGNOSIS_after"])
            ].copy()

            bracket["DIAGNOSIS"] = bracket["DIAGNOSIS_after"]

            # Keep closest exam date
            d_before = (bracket["image_date"] - bracket["EXAMDATE_before"]).abs()
            d_after  = (bracket["EXAMDATE_after"] - bracket["image_date"]).abs()
            bracket["exam_date"] = bracket["EXAMDATE_before"].where(d_before <= d_after, bracket["EXAMDATE_after"])

            # Rename and keep only relevant columns
            bracket = bracket.rename(columns={"DIAGNOSIS":"diagnosis"})
            bracket = bracket[["image_id", "image_date", "subject_id", "exam_date", "group", "modality", "diagnosis"]]
    
            
            to_concat.append(bracket)
        
        # Concatenate
        final_df = pd.concat(to_concat, ignore_index=True, sort=False)
        
        return final_df
