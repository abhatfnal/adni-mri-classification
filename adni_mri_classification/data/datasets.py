import torch
import pandas as pd
import nibabel as nib
import numpy as np
import os

from torch.utils.data import Dataset

class ADNIDataset(Dataset):
    
    def __init__(self, scan_csv, diagnostic_csv, modalities=["MRI", "amyloid-PET"], diagnosis=[1,2,3], tolerance=60):
        
        # Master scans csv
        df_scan = pd.read_csv(scan_csv)
        
        # Filter for specified modalities 
        self.df_scan = df_scan[ df_scan["modality"].isin(modalities)].copy()
        
        # Master visits csv
        self.df_diagnostic = pd.read_csv(diagnostic_csv)
        
        # Add diagnosis column to scan dataframe
        self.__add_diagnosis__(self.df_scan, tolerance)
        
        # Create multimodal samples
        
        # for patient in unique_patients:
        #       5 mri scan
        #       5 pet scan
        #       5 ecg  
        #
        #
    
    def __add_diagnosis__(self, df_scan, tolerance):
        """
        Adds diagnosis column to scan df by matching it with temporally closest visit.
        """
        
        # Ensure EXAMDATE has datetime format
        self.df_diagnostic["EXAMDATE"] = pd.to_datetime(self.df_diagnostic["EXAMDATE"])
        
        entries = []

        for index, row in df_scan.iterrows():
            
            subj_id = row["subject_id"]
            image_date = row["image_date"]
            
            # Get subject visits
            subj_visits = self.df_diagnostic[ self.df_diagnostic["PTID"] == subj_id].copy()
            
            # Compute days difference
            subj_visits["diff_days"] =  (subj_visits["EXAMDATE"] - image_date).dt.days
            
            # Choose only those under the tolerance
            match = subj_visits[ subj_visits["diff_days"].abs() <= tolerance]
            
            # If visit under tolerance is found, assign diagnosis of closest visit
            if not match.empty:
                
                # Get temporally closest visit
                visit = match.loc[match["diff_days"].abs().idxmin()]
                
                # Append image id and diagnosis to entries
                entries.append({
                    "image_id":row["image_id"],
                    "diagnosis":visit["DIAGNOSIS"]
                })
                
            # Otherwise, try what follows:
            else:
        
                # Separate visits before and after 
                subj_visits_before = subj_visits[ subj_visits["diff_days"] < 0]
                subj_visits_after = subj_visits[ subj_visits["diff_days"] >= 0]
                
                # Make sure that there are both visits before and after 
                if subj_visits_after.empty or subj_visits_before.empty:
                    continue
                
                # Turn days difference into absolute values
                subj_visits_before = subj_visits_before.copy()
                subj_visits_before["diff_days"] = subj_visits_before["diff_days"].abs()
                
                # Get closest visit before
                visit_before = subj_visits_before.sort_values("diff_days").iloc[0]
                
                # Get closest visit after 
                visit_after = subj_visits_after.sort_values("diff_days").iloc[0]
                
                # If diagnosis has not changed between the two visit, then 
                # at the time of the scan (in between the visits) the diagnosis must agree.
                if visit_after["DIAGNOSIS"] == visit_before["DIAGNOSIS"]:
                    
                    entries.append({
                        "image_id":row["image_id"],
                        "diagnosis":visit_after["DIAGNOSIS"]
                    }) 
                    
        # Create dataframe to attach to MRI df
        diagnosis_col = pd.DataFrame(entries)

        # Merge 
        df_scan = pd.merge(df_scan, diagnosis_col, on="image_id")
        
    def __create_multimodal_samples__(self, df_scan, tolerance):
        
        # Unique patients 
        patient_ids = df_scan["subject_id"].unique()
        
        # Iterate over unique patients 
        for subj_id in patient_ids:
        
            # Get scans of patient
            patient_scans = df_scan[ df_scan["subject_id"] == subj_id].copy()
            
            # Begin clustering
            
            # Get all available modalities for this patient 
            modalities = patient_scans["modality"].unique()
        
    
    
class TransformDataset(Dataset):
    """
    Simple class that applies a transform to a base dataset
    """
    def __init__(self, base_ds, transform=None):
        
        self.base = base_ds
        self.transform = transform
        
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.transform is not None:
            x = self.transform(x)
            
            return x, y
        
        
class ADNIDataset2(Dataset):
    """
    ADNI dataset class
    """
        
    def __init__(self, csv_file, classes=[1,2,3], transform=None):
        
        self.transform = transform
        
        # Read csv
        self.data = pd.read_csv(csv_file)
        
        # Only keep rows with existing files
        self.data = self.data[ self.data['filepath'].apply( lambda x : os.path.exists(x))]
        
        # Only keep requested diagnoses
        self.data = self.data[self.data['diagnosis'].isin(classes)].copy()

        # Map them to 0, ... , |classes|
        diag_to_label = {diag: i for i, diag in enumerate(classes)}

        self.data['label'] = self.data['diagnosis'].map(diag_to_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vol = nib.load(row['filepath']).get_fdata().astype(np.float32)
   
        img = torch.from_numpy(vol).unsqueeze(0)  # → (1,D,H,W)

        if self.transform:
            img = self.transform(img)  # Tensor→Tensor on same device

        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img, label
    
    def labels(self):
        return self.data['label'].astype(int).tolist()
    
    def groups(self):
        return self.data['ptid'].astype(str).tolist()
    
if __name__ == "__main__":
    
    dataset = ADNIDataset(
        csv_file="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_mri_pet/datasets/dataset_unimodal_mri_trainval.csv")
    
    index = 20
    print(dataset[index][0].shape)