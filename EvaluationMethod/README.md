# WAPE Evaluation Framework

## Getting Started
We have prepared a number of scripts: `run_hypertune.bat` and `tuned_run.bat`. These were used to hyper-parameter tune models individuallly and run multiple tests of the same run (to collect error bars).

If you followed the [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio/) instructions on downloading the conda environment you can run these as well as the following command.

```
ns-train [nerfacto/instant-ngp/mipnerf]
```

To run the tuned models the following can be used where `%%d` is the folder path which by default it `data/GT/`. This folder should contain the pregenerated data from our data synthesis method.

```
ns-train mipnerf --pipeline.datamanager.path=%%d --pipeline.datamanager.train-batch-size=4096 --optimizers.fields.optimizer.lr=5e-4 --pipeline.datamanager.loss-function-rgb=MSELoss --pipeline.datamanager.loss-function-density=L1Loss
ns-train nerfacto --pipeline.datamanager.path=%%d --pipeline.datamanager.train-batch-size=4096 --optimizers.fields.optimizer.lr=1e-2 --pipeline.datamanager.loss-function-rgb=MSELoss --pipeline.datamanager.loss-function-density=L1Loss
ns-train instant-ngp --pipeline.datamanager.path=%%d --pipeline.datamanager.train-batch-size=256 --optimizers.fields.optimizer.lr=1e-4 --pipeline.datamanager.loss-function-rgb=MSELoss --pipeline.datamanager.loss-function-density=L1Loss
```

Note that CLI commans for Nerfstudio will not work as we have modified the framework.

We have also provided the hotdog and cubeworld datasets in `data/GT_hotdog/` and `data/GT_cubeworld/`

## Differences with NeRFStudio
The Nerfstudio repository contains a lot of helpful functionality and customisation for processing NeRF data and customisation.
    
Custom Classes:
    
1. `GTDatamanger` in `nerfstudio/data/datamanagers/gt_datamanager.py` loads the pre-generated data, handles a step through the network and provides the loss function. We have also used the binary accuracy (threshold of 0.001), mean error and PSNR metrics on output parameters as the loss function alone is not good enough to indicate learning. The Nerfstudio `DataManager` class is a parent.

2. `GTPipeline` in `nerfstudio/pipelines/gt_pipelines.py` handles the the execution of training and testing steps. The `get_eval_image_metrics_and_images()` method handles the testing of novel views (PSNR, SSIM, LPIPS and WAPE) and builds the images.

## Modifications (in accordance with Apache License 2.0 held by Nerfstudio):

1. `nerfstudio/configs/method_configs.py` - this holds the main configuration file for identifying which methods are avaliable for execution and what their default parameters are. This also instantiates the methods. We have removed all models which have not been adapted.

2. `RaySamples.get_weights()` in `nerfstudio/cameras/rays.py` has been modified to remove the delta variable from the generation of weights for the rendering equation (as disucssed in the paper).

3. `nerfstudio/models/[mipnerf_custom.py, instan_ngp_custom.py, nerfacto_custom.py, vanilla_nerf_custom.py]` are modified versions of the `mipnerf, instang-ngp, nerfacto` and `vanilla nerf` implementations. We have modified the class inputs and the `get_outputs()` method to ingest `RaySamples` rather than `RayBundles` as sampling is performed outside of the class. This removes the need for pre-processing samples. We have also removed the metrics and loss functions (found in the `GTDatamanager`).

4. `nerfstudio/engine/trainer.py` - we have commented out all the writing functions which either interfere or are unecessary for our training framework.

# Custom Models
First write you model as a NeRFStudio Model (with `Model Config`). Then we need to change the following functions: 

- `fields.get_outputs` and `fields.get_outputs` : These may need to be modified to support scene with no camera-related or sampling functions.
- `model.get_outputs(rb:RayBundle)` : This should (1) ingest a batch of `RaySamples` instead, (2) perform a forward pass, (2) return a dict containing `dictkeys('rgb_fields', 'density_fields')`, which are the colour and density predictions of the model.
