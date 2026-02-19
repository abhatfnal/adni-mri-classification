


# import os
# import nibabel as nib
# import numpy as np

# from tempfile import TemporaryDirectory
# from nipype.interfaces.spm import Realign, Normalize12

# from pathlib import Path
# from scipy.ndimage import zoom

# from fsl.wrappers import bet, flirt, LOAD
# from spm import *

# from nipype.interfaces.spm import Realign, Normalize12
# from functools import partial


# #====|| Configure SPM / MATLAB paths ||====

# SPM_DIR = '/project/aereditato/cestari/spm/spm'
# MATLAB_CMD = 'matlab -nodesktop -nosplash'

# #==========================================

# # Preprocessing operations
# def fsl_bet(img,**kwargs):
#     """
#     Performs skull stripping using FSL BET.
    
#     Parameters
#     ----------
#     img: nib image
#         Source image.
        
#     Returns
#     -------
#     Processed nibabel image.
#     """
#     try:
#         return bet(img, LOAD)['output']
#     except Exception as e:
#         print(f"Error: {e}")
#         return None


# def fsl_flirt(img, template_path, dof=12,**kwargs):
#     """
#     Performs registration using FSL FLIRT
    
#     Parameters
#     ----------
#     img: nib imahe
#         Source image.
#     template_path: str
#         Absolute path to registration template.
#     dof: int
#         Degrees of freedom.
    
#     Returns
#     -------
#     Processed nibabel image.

#     """
#     try:
#         return flirt(
#             src=img,
#             ref=template_path,
#             out=LOAD,
#             dof=dof
#         )['out']
#     except Exception as e:
#         return None

# def spm_realign(img, matlab_cmd=MATLAB_CMD, spm_dir=SPM_DIR, **kwargs):
#     """
#     Perform SPM Realign on a nibabel image, returning a nibabel image.
    
#     Parameters
#     ----------
#     img : nibabel.Nifti1Image
#         Source image to realign.
#     matlab_cmd : str
#         Command to invoke MATLAB (e.g. 'matlab -nodesktop -nosplash').
#     spm_dir : str
#         Path to your SPM directory.
#     **kwargs
#         Any additional Realign inputs (e.g. quality, fwhm, etc.).
        
#     Returns
#     -------
#     Processed nibabel image.
#     """
#     with TemporaryDirectory() as workdir:
#         # 1) Dump the input image
#         in_file = os.path.join(workdir, "in.nii")
#         nib.save(img, in_file)

#         # 2) Prepare the realigner
#         realigner = Realign()
#         realigner.inputs.in_files   = in_file  # single string is OK
#         realigner.inputs.matlab_cmd = matlab_cmd
#         realigner.inputs.paths      = spm_dir
#         for k, v in kwargs.items():
#             setattr(realigner.inputs, k, v)

#         # 3) Run it *inside* the temp directory*
#         cwd = os.getcwd()
#         try:
#             os.chdir(workdir)
#             res = realigner.run()
#         finally:
#             os.chdir(cwd)

#         # 4) Load fully into memory
#         out_path = res.outputs.realigned_files
#         img2     = nib.load(out_path)
#         data     = img2.get_fdata()
#         affine   = img2.affine

#     # 5) workdir (and all scripts) auto‐removed here
#     return nib.Nifti1Image(data, affine)


# def spm_normalize12(img, matlab_cmd=MATLAB_CMD, spm_dir=SPM_DIR, **kwargs):
#     """
#     Perform SPM Normalize12 on a nibabel image, returning a nibabel image.
    
#     Parameters
#     ----------
#     img : nibabel.Nifti1Image
#         Source image to realign.
#     matlab_cmd : str
#         Command to invoke MATLAB (e.g. 'matlab -nodesktop -nosplash').
#     spm_dir : str
#         Path to your SPM directory.
#     **kwargs
#         Any additional Realign inputs (e.g. quality, fwhm, etc.).
    
#     Returns
#     -------
#     Processed nibabel image.
#     """
#     with TemporaryDirectory() as workdir:
#         in_file = os.path.join(workdir, "in.nii")
#         nib.save(img, in_file)

#         normalizer = Normalize12()
#         normalizer.inputs.image_to_align = in_file
#         normalizer.inputs.matlab_cmd     = matlab_cmd
#         normalizer.inputs.paths          = spm_dir
#         for k, v in kwargs.items():
#             setattr(normalizer.inputs, k, v)

#         cwd = os.getcwd()
#         try:
#             os.chdir(workdir)
#             res = normalizer.run()
#         finally:
#             os.chdir(cwd)

#         out_path = res.outputs.normalized_image
#         img2     = nib.load(out_path)
#         data     = img2.get_fdata()
#         affine   = img2.affine

#     return nib.Nifti1Image(data, affine)

# def load_nifti(path):
#     """
#     Loads NIfTI image from specified path.
    
#     Parameters
#     ----------
#     path: str
#         Absolute path of input .nii / .nii.gz file.
        
#     Returns
#     -------
#     nibabel image
#     """
#     try:
#         return nib.load(path)
#     except Exception as e:
#         print(f"Error: {e}")

# def to_numpy(img):
#     """
#     Converts NIfTI image to numpy array
#     """
#     try:
#         return img.get_fdata()
#     except Exception as e:
#         print(f"Error: {e}")
#         return None 


# def resize(arr, output_shape=(128,128,128), order=1):
#     """
#     Resize numpy array to desired shape
    
#     Parameters
#     ----------
    
#     arr: numpy.array
#         Source array.
#     output_shape: tuple
#         Desired output shape.
#     order: int
#         Order of interpolation: linear=1, quadratic=2,...
        
#     Returns
#     -------
#     reshaped array
#     """
#     try:
#         factors = [o/s for o, s in zip(output_shape, arr.shape)]
#         return zoom(arr, factors, order=order)
#     except Exception as e:
#         print(f"Error: {e}")
#         return None


# def zscore_norm(arr):
#     """
#     Performs Z-score normalization on input numpy array.
    
#     Parameters
#     ----------
#     arr: numpy array
#         Input array.
    
#     Returns
#     -------
#     Processed numpy array.
#     """
#     try:
#         return (arr - arr.mean()) / arr.std()
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# # Operations map
# OP_MAP = {
#     'fsl_bet':       fsl_bet,
#     'fsl_flirt':     fsl_flirt,
#     'spm_realign':   spm_realign,
#     'spm_normalize': spm_normalize12,
#     'to_numpy':      to_numpy,
#     'resize':        resize,
#     'zscore_norm':   zscore_norm,
#     'load_nifti':    load_nifti,
# }

# # Pipeline builder
# def build_pipeline(cfg):
#     """
#     Build a sequential pipeline runner from a config dict.

#     Parameters
#     ----------
#     cfg : dict or OrderedDict
#         Maps operation names to parameter dicts, e.g.
#         {
#             'fsl_bet':       {'suffix': '_brain'},
#             'spm_realign':   {},
#             'spm_normalize': {},
#             'to_numpy':      {},
#             'resize':        {'output_shape': (128,128,128), 'order': 1},
#             'zscore_norm':   {},
#         }

#     Returns
#     -------
#     runner : function(initial) -> final_value
#         A function that takes an initial filename or array and applies each step.
#     """
#     pipeline = []
#     for op_name, params in cfg.items():
#         fn = OP_MAP.get(op_name)
#         if fn is None:
#             raise ValueError(f"Unknown operation: {op_name}")
#         pipeline.append(partial(fn, **params))

#     def runner(initial):
#         val = initial
#         for step in pipeline:
#             val = step(val)
#         return val

#     return runner

# if __name__ == '__main__':
#     import argparse
#     import yaml
#     import pandas as pd
#     import numpy as np
#     import nibabel as nib
#     import os
#     import subprocess
#     import sys
#     import textwrap
#     from pathlib import Path
#     from concurrent.futures import ThreadPoolExecutor, as_completed
#     from functools import partial

#     parser = argparse.ArgumentParser(
#         description="Run a configurable preproc pipeline on a list of NIfTIs"
#     )
#     parser.add_argument(
#         '--config', '-c',
#         required=True,
#         help="YAML file with keys: csv_file, output_dir, pipeline"
#     )
#     parser.add_argument(
#         '--workers', '-w',
#         type=int,
#         default=16,
#         help="Number of parallel threads (default: 16)"
#     ),
#     parser.add_argument(
#         '--job',
#         action='store_true',
#         help="Submit SLURM job instead of running locally"
#     )
    
#     args = parser.parse_args()
    
#     # Get absolute path of configuration file
#     args.config = os.path.abspath(args.config)
    
#     # 1) load config immediately, so we know out_dir before branching
#     with open(args.config) as f:
#         cfg = yaml.safe_load(f)
        
#     csv_file     = cfg.get('csv_file')
#     out_dir      = cfg.get('output_dir')
#     pipeline_cfg = cfg.get('pipeline')
    
#     if not csv_file or not out_dir or not pipeline_cfg:
#         raise ValueError("YAML must define: csv_file, output_dir, pipeline")

#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # 2) if --job, emit one SLURM job for the *entire* pipeline
#     if args.job:
#         # where to put the sbatch wrapper script
#         script_dir = out_dir / "sbatch_scripts"
#         script_dir.mkdir(exist_ok=True)

#         this_script = Path(__file__).absolute()
#         sbatch = f"""\
#         #!/bin/bash
#         #SBATCH --job-name=preproc_%j
#         #SBATCH --output={out_dir}/preproc_%j.out
#         #SBATCH --error={out_dir}/preproc_%j.err
#         #SBATCH --partition=caslake
#         #SBATCH --ntasks-per-node=1
#         #SBATCH --cpus-per-task=8
#         #SBATCH --account=pi-aereditato
#         #SBATCH --chdir={out_dir}

#         source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
#         conda activate adni_rcc 

#         module load matlab
        
#         export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 
        
#         python {this_script} \\
#           --config "{args.config}" \\
#           --workers {args.workers}
#         """
#         sbatch = textwrap.dedent(sbatch)

#         script_file = script_dir / "run_preproc_all.sbatch"
#         script_file.write_text(sbatch)

#         subprocess.run(["sbatch", str(script_file)], check=True)
#         print(f"[SUBMITTED] {script_file}")
#         sys.exit(0)


    
#     df = pd.read_csv(csv_file)
#     if 'filepath' not in df.columns or 'diagnosis' not in df.columns:
#         raise ValueError("CSV must have both 'filepath' and 'diagnosis' columns")

#     # 3) ensure output dir exists
#     os.makedirs(out_dir, exist_ok=True)

#     # 4) build your pipeline runner
#     runner = build_pipeline(pipeline_cfg)

#     # 5) prepare to collect summary records
#     summary = []

#     # 6) worker function now takes (path, diagnosis)
#     def process_one(nii_path, diagnosis):
#         out = runner(nii_path)
#         stem = Path(nii_path).stem

#         # decide how to save
#         if isinstance(out, nib.Nifti1Image):
#             dest = Path(out_dir) / f"{stem}.nii"
#             nib.save(out, str(dest))
#         elif isinstance(out, np.ndarray):
#             dest = Path(out_dir) / f"{stem}.npy"
#             np.save(str(dest), out)
#         else:
#             src  = Path(out)
#             dest = Path(out_dir) / src.name
#             os.replace(src, dest)

#         return str(dest), diagnosis

#     # 7) run in parallel, carrying diagnosis through
#     with ThreadPoolExecutor(max_workers=args.workers) as exe:
#         futures = {
#             exe.submit(process_one, row['filepath'], row['diagnosis']): row
#             for _, row in df.iterrows()
#         }
#         for fut in as_completed(futures):
#             row = futures[fut]
#             try:
#                 dest_path, diag = fut.result()
#                 print(f"[OK]   {row['filepath']} → {dest_path}")
#                 summary.append({'filepath': dest_path, 'diagnosis': diag})
#             except Exception as exc:
#                 print(f"[FAIL] {row['filepath']} → {exc}")

#     # 8) write out the summary CSV
#     summary_df = pd.DataFrame(summary)
#     summary_csv = Path(out_dir) / "processed_with_labels.csv"
#     summary_df.to_csv(summary_csv, index=False)
#     print(f"\nWrote summary CSV with labels to:\n  {summary_csv}")
