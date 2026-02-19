#!/usr/bin/env python3
import os
from nipype import Node, Workflow
from nipype.interfaces import spm
from nipype.interfaces.spm import SliceTiming, Realign, Normalize12
from nipype.interfaces.fsl import BET


def preprocess(path):
    pass 

# ───────────  User settings ────────────
input_file = '/project/aereditato/cestari/ADNI/merged/ADNI/002_S_0729/MPR____N3__Scaled/2006-08-02_07_02_00.0/I40692/ADNI_002_S_0729_MR_MPR____N3__Scaled_Br_20070217001301848_S17535_I40692.nii'       # your 4D .nii
spm_dir    = os.path.expanduser('/project/aereditato/cestari/spm')   # where you unpacked SPM12
template   = os.path.join('/project/aereditato/cestari/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz')

# 1) configure SPM to use your MATLAB module
spm.SPMCommand.set_mlab_paths(
    matlab_cmd='matlab -nodesktop -nosplash',
    use_mcr=False,
    paths=[spm_dir]
)

realign = Realign()

realign.inputs.in_files = 'functional.nii'
realign.inputs.register_to_mean = True
realign.run() 

bet = BET()
bet.inputs.in_file = input_file
bet.inputs.out_file = "./T1w_nipype_bet.nii"
res = bet.run()


