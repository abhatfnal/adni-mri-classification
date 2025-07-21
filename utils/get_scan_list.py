"""
Given .zip ADNI archive with DICOM images, returns
list of unique MRI scans.

"""

import os
import sys
import argparse
import zipfile

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('path', help='Path to .zip archive')
    
    args = parser.parse_args(sys.argv[1:])
    
    try:
        
        zip = zipfile.ZipFile(args.path)

        # Get paths of files inside the archive
        filepaths = zip.namelist()

        # Get names (e.g. XXXX.dcm)
        filenames = [ os.path.basename(p) for p in filepaths if '.dcm' in p]

        # Get unique scan ids
        scan_ids = set([ name.split('_')[-1].split('.dcm')[0] for name in filenames])

        # Print them to console
        for id in scan_ids:
            print(id)
            
    except Exception as e:
        print(f"Error: {e}")