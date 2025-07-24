"""
Given .zip ADNI archive with DICOM images, returns
list of unique MRI scans.

"""

import os
import sys
import argparse
import zipfile

def get_scan_list(path):
    
    try:
        zip = zipfile.ZipFile(path)

        # Get paths of files inside the archive
        filepaths = zip.namelist()

        # Get unique scan paths
        scan_ids = list(set([ "/".join(path.split("/")[:5]) + "/" for path in filepaths if ".dcm" in path]))

        return sorted(scan_ids)
            
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('path', help='Path to .zip archive')
    
    args = parser.parse_args(sys.argv[1:])
    
    for id in get_scan_list(args.path):
        print(id)
    
    