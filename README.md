# Patch-Conv
Evaluating different ways of convolving on batches of patches

## Dependencies
PyTorch 1.2

## To use

Install with: `python setup.py install`

To run the profile scripts: `python profile_comparison.py`

You can check that the custom CUDA kernels for `patch_im2col` and `patch_col2im` and  are correct by running `python test_patch_conv.py` and/or `python test_transposed_patch_conv.py`.