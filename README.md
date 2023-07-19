
# Instructions for using NeRF method

1. `BlenderDataExtraction` - Use this to extract Blender data and remeber to change the directory to place the resulting file. We advise directly placing it into the `DataSynthesis` folder.

2. `DataSynthesis` - Handles the pre-generation of synthetic data. Once performed, remeber to move the contents of `DataSynthesis/save_data/` to `EvaluationMethod/data/`. By default the evaluation method reads from `EvaluationMethod/data/GT/` so it's advised to place filed in there. We have provided `GT_no_reflection` and `GT_reflection` which contains the pregenerated data from our test scene. We have also provided `GT_hotog` which models hotdog with a mirror-like plate.

3. `EvaluationMethod` - Handles the training and evaluation of the WAPE metric and PSNR, SSIM and LPIPs comparisons. You can use `ns-train -h` to show the avaliable methods to execute.

## Instalation/Dependencies

1. Follow the [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio/) instructions for installation. Do not download nerfstudio with pip (we have not tested it), build from source.

2. We used additionally require `tensordict==0.0.2b0, tdqm`. Ensure tensor dict is not more recent as we exploit a bug in the old version to initialise empty tensor dictionaries.

## Notes
We last updated the Nerfstudio packages on 13/04/2023.

In each folder we attach a `README.md` which will guide you through each step in the framework. Additionally we have provided the detail on our use of the Nerfstudio repository in each folder. Under the Apache License 2.0, modifications of each use of the Nerfstudio repository have been noted in the readme files associated with the different uses of the repository.

If you want to run a custom scene, instructions are provided on setting up Blender in `BlenderDataExtraction/extract_data.py` which should be run in the Blender Scripting API. Otherwise you are free to modify the provided Blender files. Note that this implementation has only been tested on an Nvidia RTX 3090.

If you encounter any errors with running the code: Check MSVS version, Cuda Kernels have the right version (tinycudann dependency requires it). If the code itself has errors, download the nerfstudio repo and paste-in the contents of the relevant folder.

# Instructions for using Depth INR method

1. The folder contains the simple implementation of a cuboid scene and does not depend on extrnal data. It pregenerates data though everything can is executed under the same command.

2. Ihe `README.md` file contains installation, customisation and explanation.