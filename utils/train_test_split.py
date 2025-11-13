"""
Splits dataset into train+val and test, producing trainval.csv and test.csv file
in the desired output directory. To be run once at the very beginning of experimentation, 
after preprocessing.
"""
import os
import sys
import pandas as pd
import argparse

# from sklearn.model_selection import train_test_split  # no longer used
from sklearn.model_selection import StratifiedGroupKFold

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""Splits dataset into train+val and test, producing trainval.csv and test.csv file
        in the desired output directory. To be run once at the very beginning of experimentation, 
        after preprocessing.""")
    
    parser.add_argument('--input-csv', required=True, help='Path to dataset csv file.')
    parser.add_argument('--test-size', required=True, help='Size of test set as ratio.')
    parser.add_argument('--output-dir', required=True, help='Output dir where trainval.csv and test.csv are dumped.')
    parser.add_argument('-r', required=False, help='Random state of the split')
    
    args = parser.parse_args(sys.argv[1:])
    
    try:
        # Assert that test size is valid
        assert float(args.test_size) > 0 and float(args.test_size) < 1, "Test size must be between 0 and 1 (exclusive)"
        
        # Read dataset csv
        df = pd.read_csv(args.input_csv)
        
        # Get labels and groups
        labels = df['diagnosis'].astype(int).tolist()
        groups = df['rid'].astype(str).tolist()
        
        # Get random state if specified
        random_state = int(args.r) if args.r else 42
        
        # Use StratifiedGroupKFold to keep groups together and preserve class balance
        # Approximate test_size by choosing one fold as test
        test_size = float(args.test_size)
        n_splits = max(2, int(round(1.0 / test_size)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Take the first fold as test
        test_indices = None
        for _, test_idx in sgkf.split(df.index, y=labels, groups=groups):
            test_indices = test_idx
            break
        
        # Trainval and test dataframes
        mask = pd.Series(False, index=df.index)
        mask.iloc[test_indices] = True
        
        trainval_df = df[~mask].reset_index(drop=True)
        test_df = df[mask].reset_index(drop=True)
        
        # Dump dataframes to .csv
        os.makedirs(args.output_dir, exist_ok=True)
        trainval_df.to_csv(os.path.join(args.output_dir, 'trainval.csv'), index=False)
        test_df.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
        
    except Exception as e:
        print(f"Error: {e}")
