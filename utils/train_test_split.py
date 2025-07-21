"""
Splits dataset into train+val and test, producing trainval.csv and test.csv file
in the desired output directory. To be run once at the very beginning of experimentation, 
after preprocessing.
"""
import os
import sys
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('csv_path', help='Path to dataset csv file.')
    parser.add_argument('test_size', help='Size of test set as ratio.')
    parser.add_argument('out_dir', help='Output dir where trainval.csv and test.csv are dumped.')
    parser.add_argument('-r',required=False, help='Random state of the split')
    
    args = parser.parse_args(sys.argv[1:])
    
    try:
        
        # Assert that test size is valid
        assert float(args.test_size) >= 0 and float(args.test_size) <= 1, "Test size must be between 0 and 1"
        
        # Read dataset csv
        df = pd.read_csv(args.csv_path)
        
        # Get labels 
        labels = df['diagnosis'].astype(int).tolist()
        
        # Get random state if specified
        if args.r:
            random_state = int(args.r)
        else:
            random_state = None
        
        # Split using specified random state
        trainval_indices, test_indices = train_test_split(
            list(range(len(labels))),
            test_size=float(args.test_size),
            random_state=random_state,
            stratify=labels)
        
        # Trainval dataframe
        trainval_df = df.iloc[ trainval_indices ].reset_index(drop=True)
        
        # Test dataframe
        test_df = df.iloc[ test_indices ].reset_index(drop=True)
        
        # Dump dataframes to .csv
        trainval_df.to_csv( os.path.join(args.out_dir, 'trainval.csv'), index=False)
        test_df.to_csv( os.path.join(args.out_dir, 'test.csv'), index=False)
        
    except Exception as e:
        print(f"Error: {e}")
