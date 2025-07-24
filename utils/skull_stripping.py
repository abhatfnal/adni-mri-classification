"""
Applies skull stripping to input .nii file.

"""
import os
import sys
import argparse
from fsl.wrappers import bet

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Applies skull stripping to .nii file')
    
    parser.add_argument("input_file",help="Input file path")
    parser.add_argument("output_file",help="Output file path")
    
    args = parser.parse_args(sys.argv[1:])
    
    #Set fsl output type
    os.environ["FSLOUTPUTTYPE"] = "NIFTI"
    
    try:
        bet(args.input_file, args.output_file)
    except Exception as e:
        print(f"Error: {e}")