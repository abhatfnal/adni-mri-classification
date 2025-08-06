import torchio as tio

# Path to your input and output files
input_path = './test.nii'
output_path = './test_flipped.nii'


# 1) Load the volume as a TorchIO image
image = tio.ScalarImage(input_path)

# 2) Build the RandomFlip transform with your probability
flip = tio.RandomFlip(axes=('LR',), p=0.5)

# 3) Apply the transform (it will flip with probability p, else return original)
flipped_image = flip(image)

# 4) Save the result to disk
flipped_image.save(output_path)

print(f"Saved flipped version to {output_path}")

