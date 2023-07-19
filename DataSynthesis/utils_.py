"""General utilities for 
"""
import torch
from typing import Optional

from nerfstudio.model_components.ray_samplers import Sampler
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.cameras.rays import RayBundle as RB

"""Define custom ray bundles (containing only origins and directions)

    Note:
        These are different to NeRFstudio raybundles!
"""
class RayBundle():
    origins: torch.Tensor
    directions: torch.Tensor

    def __init__(self, origins, directions):
        self.origins = origins
        self.directions = directions

"""Define the plane of a cuboid in our scene for N surfaces

    Properties:
        bounds: Tensor, shape(N, 4,3) containing the coordinates of the planes corners
        normals: Tensor, shape(N, 3) containing the normals of each plane
        colours: Tensor, shape(N, 3) containing the colours of each surface
        opacities, Tensor, shape(N, 1) containing the opacities of each surface
"""
class Surfaces:
    def get_surf_data(self, id):
        """Using the surface index return the surface data

            Args:
                id, int, index of surface in the stack

            Notes:
                This is used for display (mayavi.mlab library is used)
        """
        assert id < len(self.colours), f'ID {id} not valid'
        assert self.bounds is not None, 'Need to load scene !!'

        b = self.bounds[id]
        col = tuple(self.colours[id])
        opa = self.opacities[id]
        x = []
        y = []
        z = []
        traingles = []
        i1 = 0
        i2 = i1 + 1
        i3 = i1 + 2
        x = x + list(b[:3,0].cpu().numpy())
        y = y + list(b[:3,1].cpu().numpy())
        z = z + list(b[:3,2].cpu().numpy())
        traingles.append((i1, i2, i3))

        textx = (b[0,0].cpu().numpy() +b[1,0].cpu().numpy()+b[2,0].cpu().numpy())/3.
        texty = (b[0,1].cpu().numpy() +b[1,1].cpu().numpy()+b[3,1].cpu().numpy())/3.
        textz = (b[0,2].cpu().numpy() +b[1,2].cpu().numpy()+b[3,2].cpu().numpy())/3.
        return x,y,z, traingles, col, opa, textx, texty, textz
            

    def __init__(self, 
        bounds:torch.Tensor, 
        normals:torch.Tensor,
        colours:list,
        opacities: list):        
        self.num_surfaces = len(colours)
        self.bounds = bounds
        self.normals = normals
        self.colours = colours
        self.opacities = opacities


"""Sample Explicit Points

    Args:
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    
    Notes:
        Uses the nerfstudio Sampler class and modifies the ray-generation from raybundle data
"""
class ExplicitSampler(Sampler):
    

    def __init__(
        self,
    ) -> None:
        super().__init__()
    
    def generate_ray_samples(
        self,
        bin_starts=None,
        bin_ends=None,
        ray_bundle: Optional[RB] = None,
        num_samples: Optional[int] = None,
        
    ) -> RaySamples:
        """Generates position samples according to spacing function.

        Args:
            ray_bundle: Rays to generate samples for
        Returns:
            Ray Samples
        """
        assert ray_bundle is not None
        assert bin_starts is not None and bin_ends is not None
       
        bs = bin_starts.unsqueeze(0)
        be = bin_ends.unsqueeze(0)
        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=bs,
            bin_ends=be
        )
        return ray_samples

def get_normals(V):
    """Given a batch of traingular surfaces compute the normals
    
        Args
            V: torch.Tensor, shape (..., M, 3), where M is the number of vertices where (M > 3)  

        Return
            N: torch.tensor, shape (..., 3) UN-NORMALISED!! This is a norm normal direction vector
    """
    e1 = V[..., 1, :] - V[..., 0, :]
    e2 = V[..., 2, :] - V[..., 0, :]
    m = e2.cross(e1)
    return m

def normalise_normals(n):
    """ Normalise Tensor n
    """
    return n / torch.linalg.norm(n)


def get_reflection_direction(dir, norm):
    """Get the reflected directions of a batch of rays
        Args:
            dir, Tensor, ray directions tensor
            norm, Tensor, surface normals tensor
        
        Return:
            d - 2((d.n)/(n.n))*n, Tensor, reflected ray
    """
    return dir - 2.* (torch.diagonal(torch.mm(dir,norm.T))/ torch.diagonal(torch.mm(norm,norm.T))).unsqueeze(-1)*norm
