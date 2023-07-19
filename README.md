# Towards a Robust Framework for NeRF Evaluation

This repository contains the code release for [Towards a Robust Framework for NeRF Evaluation](https://arxiv.org/abs/2305.18079). The contents of the repository relate to the experiments in the paper. Thus, we present exemplar implementations of frameworks for seperately evaluating NeRFs and INRs using explicit radiance field representations. The code should be treated as **example** implementations of our framework - see the limitations section in the paper to understand why.

For NeRF Evaluations we use data extracted from Blender. This is done by running the python script in the `BlenderDataExtraction` folder into the Blender Python API to extract relevant data. Instructions for use are provided within the script and can be viewed in Blender. Notes for use: The implementation is a proof of concept so evalation code is not optimised for complex scenes (with many cameras and polygons) so bare this in mind when testing.

For INR Evaluations we tested a very simple scene (three cubes and three views) to ensure tested models attain optimum performance for evaluations. Due to its simplicity, we provide a function for data generation which is exectuted before each run. 

## Requirements
<details>
<summary>Requirements</summary>

We ran this on an Nvidia RTX 3090 with 64GB of RAM with CUDA 11.8. Refer to sections below for more information of dependencies.

</details>

## Notes for transparency
<details>
<summary>Transparency</summary>

1. In the initial arXiv submission we miscalculated the PSNR results for the depth INR experiment. Additionally, for the NeRF experiment we forgot to divide the PSNR, SSIM and LPIPS results by the batch size. We have since updated the results accordingly and note that our conclusions remain the same. 

2. For the depth INR experiment it is worth noting that results (particularly for SIREN) are sensitive to initialisation. While our results do not cohere with the results from [WIRE: Wavelet Implicit Neural Representations
](https://arxiv.org/abs/2301.05187) ([code](https://github.com/vishwa91/wire)) it is important to note that there may be redeeming factors in performance associated with more complex scenes which we have not tested for (such as those in the WIRE paper).

</details>

## Details on NeRF Evaluation
The following list indicates the relevant folders associated with data generation and evaluation of NeRFs.

1. `BlenderDataExtraction` - Use this to extract Blender data and remeber to change the directory to place the resulting file. We advise directly placing it into the `DataSynthesis` folder.

2. `DataSynthesis` - Handles the pre-generation of synthetic data. Once performed, remeber to move the contents of `DataSynthesis/save_data/` to `EvaluationMethod/data/`. By default the evaluation method reads from `EvaluationMethod/data/GT/` so it's advised to place filed in there. We have provided `GT_no_reflection` and `GT_reflection` which contains the pregenerated data from our test scene. We have also provided `GT_hotog` which models hotdog with a mirror-like plate.

3. `EvaluationMethod` - Handles the training and evaluation of the WAPE metric and PSNR, SSIM and LPIPs comparisons. You can use `ns-train -h` to show the avaliable methods to execute.

<details>
<summary>Installation</summary>

1. Follow the [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio/) instructions for installation. Do not download nerfstudio with pip (we have not tested it), build from source.

2. We used additionally require `tensordict==0.0.2b0, tdqm`. Ensure tensor dict is not more recent as we exploit a bug in the old version to initialise empty tensor dictionaries.

</details>

<details>
<summary>Notes for implementation</summary>

We last updated the Nerfstudio packages on 13/04/2023.

In each folder we attach a `README.md` which will guide you through each step in the framework. Additionally we have provided the detail on our use of the Nerfstudio repository in each folder. Under the Apache License 2.0, modifications of each use of the Nerfstudio repository have been noted in the readme files associated with the different uses of the repository.

If you want to run a custom scene, instructions are provided on setting up Blender in `BlenderDataExtraction/extract_data.py` which should be run in the Blender Scripting API. Otherwise you are free to modify the provided Blender files.

If you encounter any errors with running the code: Check MSVS version, Cuda Kernels have the right version (tinycudann dependency requires it). If the code itself has errors, download the nerfstudio repo and paste-in the contents of the relevant folder.

</details>


## Details on Depth INR Evaluation

1. The folder contains the simple implementation of a cuboid scene and does not depend on external data. It pregenerates data though everything can is executed under the same command.

2. Ihe `README.md` file contains installation, customisation and explanation.