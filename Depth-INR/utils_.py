from math import log10, sqrt

import numpy as np
import torch

"""Accuracy : Rather than taking boolean accuracy we take the average error for any prediction
"""
def PSNR(preds, n_t):
#     preds_ = torch.round(preds, decimals=6)
    n_pred = preds[:,0]
    mse = torch.pow(torch.mean((n_t - n_pred)), 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1000
    psnr = 20 * log10(max_pixel / torch.sqrt(mse))
    return psnr


def Accuracy(preds, n_t):
#     preds_ = torch.round(preds, decimals=6)
    n_pred = preds[:,0]
    
    # Sum difference accuracy
    if True:
        n_diff = torch.abs(n_pred - n_t)
        n_diff=torch.sum(n_diff)/float(n_pred.size(0))
        # return (n_diff+f_diff)/2
        return n_diff

def Accuracy_elementwise(preds, n_t):
#     preds_ = torch.round(preds, decimals=6)
    n_pred = preds[:,0]
    n_diff = torch.abs(n_pred - n_t)

    return n_diff

""" Loss function:
"""
def Loss(looser, preds, n_t):
    n_pred = preds[:,0]

    return looser(n_t, n_pred)

class RayBundle():
    origins: torch.Tensor
    directions: torch.Tensor

    def __init__(self, origins, directions):
        self.origins = origins
        self.directions = directions

"""Define the plane of a cuboid in our scene
    Properties:
        bounds: Tensor, shape(4,3) containing the coordinates of the planes corners
"""
class Planes:
    bounds:torch.Tensor 
    id:int
    def __init__(self, bounds:torch.Tensor, id=None):
        assert (bounds.shape[1] == 4 and bounds.shape[2] == 3), f"Plane has incorrect shape, must be shape (batch_size, 3, 2)"
        assert id is not None
        self.bounds = bounds
        self.id = id

"""Define ray for generation of ray-tracing equation
"""
class Ray:
    def __init__(self, scale, shift, dirs):
        self.scale = scale
        self.shift = shift
        self.dirs = dirs

"""Method for generating cuboid blocks in scene
    Args:
        size: tuple, xyz size of cube
        origins: tuple, xyz origins of Bottom LH corner of cube (in 3D this is the bottom point closest to the origin)
        cube: Tensor, Template cube shape (which we apply transformations to)

    TODO: 
        (easy) Vectorise this process (use tensors for size and origins inputs)
"""
def build_block(size, origins, cube):
    x_s,y_s,z_s = size

    cube[:,:, 0] = cube[:,:, 0]*x_s
    cube[:,:, 1] = cube[:,:, 1]*y_s
    cube[:,:, 2] = cube[:,:, 2]*z_s


    x,y,z = origins
    cube[:,:, 0] = cube[:,:, 0] + x
    cube[:,:, 1] = cube[:,:, 1] + y
    cube[:,:, 2] = cube[:,:, 2] + z

    return cube

"""Method for generating ray-views: 
    The y-z unit (square) planar surface is created and transformed to our desired parameters
    
    Args:
        y_sample, z_sample: int, sample size along y and z axis
        scale: tuple, size of view
        dir: tuple, predefined direction of plane
        shift: tuple, predefined xyz shift of plane
"""
def get_view_rays(x_sample, z_sample, scale, dir, shift):

    x_vals = np.linspace(0, scale[0], x_sample)
    z_vals = np.linspace(0, scale[2], z_sample)
    y = 0.

    os = torch.tensor([[[x,y,z] for x in x_vals] for z in z_vals]).double()
    origins = os.flatten(0,1)
    # origins = origins.flatten(-1)
    directions = torch.tensor([[0., -1, 0.] for i in range(origins.size(0))])


    thex, they, thez = dir
    Rxyz = torch.tensor(
        [ [
            np.cos(thez)*np.cos(they), -
            np.cos(they)*np.sin(thez), 
            np.sin(they)
        ], [
            np.sin(thex)*np.sin(they)*np.cos(thez)+np.cos(thex)*np.sin(thez), 
            -np.sin(thex)*np.sin(they)*np.sin(thez)+np.cos(thex)*np.cos(thez),
            -np.sin(thex)*np.cos(they)
        ],[
            -np.cos(thex)*np.cos(thez)*np.sin(they)+np.sin(thex)*np.sin(thez),
            np.cos(thex)*np.sin(they)*np.sin(thez)+np.sin(thex)*np.cos(thez),
            np.cos(thex)*np.cos(they)
        ]
        ]
    ).double()
    
    OR = torch.mm(origins, Rxyz)
    Shift = torch.tensor([[shift[0]], [shift[1]], [shift[2]]]).repeat(1,origins.size(0)).double()
    ORS = OR +Shift.T

    p1 = ORS[0]
    p2 = ORS[int(x_sample-2)]
    p3 = ORS[int((x_sample*z_sample)-z_sample+1)]

    n = torch.cross((p2-p1), (p3-p1))/10.

    rays = []
    for xs in ORS:
        rays.append([xs[0],xs[1],xs[2], n[0], n[1], n[2]])
    
    return rays
