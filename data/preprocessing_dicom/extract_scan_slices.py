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
    
    parser.add_argument('scan_dir', help='Path to scan directory (inside archive)')
    parser.add_argument('archive_path', help='Path to .zip archive')
    parser.add_argument('out_dir',help='Path to output folder')
    
    args = parser.parse_args(sys.argv[1:])
    
    
    try:
        
        # Create output dir if no existent
        os.makedirs(args.out_dir, exist_ok=True)
        
        zip_ref= zipfile.ZipFile(args.archive_path, 'r')

        for member in zip_ref.namelist():
            
            # If file is in scan path (and is not a dir)
            if member.startswith(args.scan_dir) and not member.endswith('/'):
                
                filename = os.path.basename(member)
                target_path = os.path.join(args.out_dir, filename)
            
                # Read from zip and write directly to output_dir (no folders)
                with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                    target.write(source.read())
            
    except Exception as e:
        print(f"Error: {e}")
