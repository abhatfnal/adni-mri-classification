#!/usr/bin/env python3
import argparse, nibabel as nib, matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('infile')
p.add_argument('outfile')

args = p.parse_args()
data = nib.load(args.infile).get_fdata().ravel()

plt.hist(data, bins=100)
plt.savefig(args.outfile)
