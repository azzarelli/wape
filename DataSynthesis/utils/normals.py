import torch


def get_normals(V):
    """Given a BATCH of traingular surfaces (with minimum 3 points) compute the normals
    
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
    return n / torch.linalg.norm(n)