
import os 
import re
import sys
import argparse
import shutil

def main(data_dir, delete):

    # Loop through subdirs 
    for path, dirs, files in os.walk(data_dir):
        
        if path == data_dir:
            continue
        
        image_id = path.split('/')[-1]
        complete_files_names = [
                            "clean_w_masked_m" + image_id + ".nii",         # MRI
                            "clean_w_masked_rstatic_" + image_id + ".nii"]  # PET
            
        complete = False
        for file in files:
            if file in complete_files_names:
                complete = True
        
        if not complete:

            if re.match(".*I[0-9]{2,}$",path): # Double check and make sure it's a scan dir!
                print(f"Incomplete folder: Id: {image_id} Path: {path}, artifacts: {files}")
                shutil.rmtree(path)
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
 
    parser.add_argument('--delete', action="store_true", help='Delete incomplete folders')
    parser.add_argument('data_dir', help='Path to data dir where preprocessed files are be stored')

    args = parser.parse_args(sys.argv[1:])

    main(str(args.data_dir), args.delete)


