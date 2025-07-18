"""
Given scan id and .zip archive path, extracts all .dcm slices to specified 
output folder.
"""

import os
import sys
import argparse
import zipfile
import shutil

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('id', help='Id of mri scan to extract')
    parser.add_argument('archive_path', help='Path to .zip archive')
    parser.add_argument('out_dir',help='Path to output folder')
    
    args = parser.parse_args(sys.argv[1:])
    
    try:
        
        zip_ref= zipfile.ZipFile(args.archive_path, 'r')

        # Get paths of files inside the archive
        filepaths = zip_ref.namelist()

        # Get paths of slices corresponding to mri id
        slices = [ p for p in filepaths if ( str(args.id) in p and p.endswith('.dcm'))]
        
        # Absolute out dir path
        out_dir = os.path.abspath(args.out_dir)
        
        # Extract all slices
        for slice in slices:
            
            #Get slice name
            name = os.path.basename(slice)
            
            # Get output file path
            out_filepath = os.path.join(out_dir, name)
            
            # Dump to output folder
            with zip_ref.open(slice) as source:
                
                with open(out_filepath, 'wb') as dest:
                    
                    shutil.copyfileobj(source, dest)
            
    except Exception as e:
        print(f"Error: {e}")
