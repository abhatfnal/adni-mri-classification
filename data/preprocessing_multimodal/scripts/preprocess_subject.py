import pandas as pd

def main(csv_path, subject_id,):

    # Read dataset csv
    df = pd.read_csv(csv_path)

    # Filter for scans belonging to the current patient
    df = df[df["subject_id"] == subject_id]

    # Preprocess subject's MRIs first
    # If no MRI, skip to next patient (PET need MRI for preprocessing)

    # Preprocess subject's PETs

    pass 