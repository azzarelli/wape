# Data Synthesis Method for Explicit NeRF representation

## Getting Started
We will use the `test.json` file as an example. This can be generated by extracting mesh data in the Blender Python API (in Blender). Place this file in the main folder. We have provided `cubeworld.json`, `hotdog.json` and `icosphere.json` which were used in the paper.

It will take some time to generate data...


Use the following to execute the default program:
```
python run.py --camera-shape [pixels in x-view] [pixels in y-view] --eval-views [list of views, ints]
```

For the red icosphere model we used the following command (where `-b` for batchsize),
```
python run.py -n icosphere --camera-shape 125. 125. -b 10000 512 --eval-views 21 22 23 24
```

For the hotdog model we used the command,
```
python run.py -n hotdog --camera-shape 50. 50. -b 10000 512 --eval-views 20 21 22 23
```

For the dispersed cubed model we used the command,
```
python run.py -n cubeworld --camera-shape 100. 100. -b 10000 512 --eval-views 37 38 39
```

Note that the `--camera-shape` is half the size of the final resolution. E.g. we load 250x250 and 100x100 for the icosphere and hotdog models respectively. This is necessary for using nerfstudio's camera model.

To use your own data use `--name` (or `-n`) to provide a file name. Use `-h` for help.


## Moving Data to the Training and Evaluation Environment
Once data has been generated transfer the contents of `save_data/` (not the folder) to the `EvaluationMethod/data/GT/`.

## Additional Information
Prior to generation, execute the runfile with `-cxcy 10. 10.` and verify that the views for evaluation are correct. This can be done by checking the image outputs at `save_data\images\` and noting down the image index (name) of the views you want to evaluate (i.e. novel views). These can be set with `--eval-views 3 4 19`, where images `3.png, 4.png, 19.png` represent the novel views.

You can also use `--batchsize [ray batch size] [surface batch size]` (or `-b`) if you want to configure the size of ray and surface batches which are processed. 

We have provided the pregenerated data in `save_data/` incase you over-write the test data.

## File Structure:

`run.py` - main runfile 

`generator.py` - handles the loading, running and processing of steps in our synthesis algorithm

`solver.py` - handles the solving of triangular mesh to NeRF volume geometry

`scene.py` - handles the scene construction (using geometric data) and executes the shading function

`utils_.py` - contains sampling and base classes for the modules in our data synthesis algorithm

## Nerfstudio Dependencies:
All dependencies are placed in `nerfstudio/` and only `nerfstudio/cameras/rays.py` has been modified whereby we have removed the density term from the `get_weights()` method for rendering NeRF weights field from the density field. The Licence has been followed.

In `utils_.py` we use the `RayBundle`, `RaySamples` and `Sampler` classes to define our own custom sampler.

In `generator.py` we use the `Cameras`, `CamraType` and `RGBRenderer` to define methods for camera modelling and NERF render equation, which allows us to embed distortions and model different cameras. We choose the `PERSPECTIVE` camera model a default.