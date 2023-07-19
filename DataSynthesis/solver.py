"""Handles solving the geometric transform from mesh to radiance field

    TODO: use einsum instead of matmul and reduce the number function calls (many can be combines)

    Notes:
        The issues with computation don't lie with the solver but it might be nice to optimise anyways
"""
import torch


class Solver:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self,
        max_render_distance:float=1000.
    ) -> None:
        self.max_d = max_render_distance
    
    def get_ray_plane_intersections_t(self,):
        """Find t-distance scalar for the points where rays intersect wth plane
                Warning: we are dealing with batches of planes and batches of rays all at once! If your scene is huge, this will surely kill.

            Notes:
                The planes are refered to by their a, b, c and k components, as part of the planar equation: `p(.)= ax+by+cz+k=0`.
                Additionally, ray-points are referd to by their o, d and t components, as part of the ray equation: r = o + td, where r is vector in R^3 (x,y,z)
                
                Our objective is to solve for t, where r lies along a p(.). 
                
                To do this,we first need to solve for `k`, to define the whole planar reference. We do this by finding the unit-norm (normalised cross-product, n) of two surface axis' and multiplying the unit-norm by another point on the surface to find k (diagonal function is used for matrix computation). This represents the function `k = -(x, y, z).n`

                Next: Consider, x.n+k = 0 defines a point on a plane given x and k. As we now know k, we just need to find x which satisfies this equation. "Simulataneously", we need to solve r = x, thus, we are given the equaiton (o+td).n+k = 0, thuss `o.n + t(d.n) + k = 0`. Finally, `t = - (o.n + k) / (d.n)`. Which is solved in batched form.
        """
        # Fetch the boundaries of the surface-blocks
        P = self.surface_bounds
        # # Get two arbitrary vectors along the plane (to solve planar definition)
        v1 = P[:, 1] - P[:, 0]
        v2 = P[:, 2] - P[:, 0]  
        # Solve plane's normal vector
        v = torch.cross(v1,v2, dim=1)
        # Reduce planes normal to unit-norm (divide vector by HCF)
        mag_v = torch.linalg.norm(v, dim=1).unsqueeze(1)
        v = torch.div(v, mag_v) # this should give us a,b,c in planar equation
        k = - torch.diagonal(torch.mm(v, P[:,3].T)) # get scalar position along plane normal
        
        # Solve ray-plane intersection equation
        vd = torch.mm(v, self.d.T)
        vo = torch.mm(v, self.o.T)
        k1 = k.unsqueeze(1).repeat(1,vo.size(1))
        kvo = -torch.add(k1, vo)
        t = torch.div(kvo, vd)
        return t
 
    def tansform_t_to_xyz(self, t):
        """Transform t-values to r_xyz values using ray equation
        """
        ts = t.unsqueeze(2)
        d = self.d.unsqueeze(0).repeat(ts.size(0), 1,1)
        o = self.o.unsqueeze(0).repeat(ts.size(0), 1,1)
        td = ts*d
        r = o+td
        return r

    def on_surface_filter(self, r):
        """Reduce all planar intersections to the existing surface
            
            Args:
                r: Tensor, shape(# of planes, # of rays, 3) for xyz intersections given a plane and ray crossing
            
            Return:
                on_surf_indices:, Tensor, shape (# intersections, surface id, ray id) pointer to the indexs in global matrix containing valid intersections
            Notes:
                Using Baycentric Ray-triangle Intersection, finding gama and beta then checking conditions for in-plane.
        
            TODO: 
                Rearrange equations to remove div by 0 
                Seperate alpha, beta and gam into seperate functions to remove need for 'del' command (i.e. better memory management)
        """
        # Fetch the boundaries of the surface-blocks
        P = self.surface_bounds
        
        a = P[:,0].unsqueeze(1).repeat(1, r.size(1), 1)
        b = P[:,1].unsqueeze(1).repeat(1, r.size(1), 1)
        c = P[:,2].unsqueeze(1).repeat(1, r.size(1), 1)
        e = .0000000001

        # Solve Gam Nominator and denominator
        A = ((a[:,:,0] * r[:,:,2])-(a[:,:,2] * r[:,:,0]))
        B = ((a[:,:,0]*b[:,:,2]) - (a[:,:,2] * b[:,:,0]))
        C = ((a[:,:,0] * r[:,:,1]) - (a[:,:,1] * r[:,:,0]))
        D = ((a[:,:,0] * b[:,:,1]) - (a[:,:,1] * b[:,:,0]))

        E = ((a[:,:,0]*c[:,:,2]) - (a[:,:,2]*c[:,:,0]))
        F = (((a[:,:,0]*b[:,:,2]) - (a[:,:,2]*b[:,:,0])) * ((a[:,:,1]*c[:,:,0]) - (a[:,:,0]*c[:,:,1])))
        G = ((a[:,:,0]*b[:,:,1]) - (a[:,:,1]*b[:,:,0]))

        zeros = (D == 0.).nonzero(as_tuple=True)
        D[zeros[0], zeros[1]] = e
        zeros = (G == 0.).nonzero(as_tuple=True)
        G[zeros[0], zeros[1]] = e

        g_nom = A - ((B * C) / D)
        g_denom = E + (F / G)
        zeros = (g_denom == 0.).nonzero(as_tuple=True)
        g_denom[zeros[0], zeros[1]] = e
        gam = g_nom / g_denom
        # print(A, ((B * C) / D))

        del A,B,C,D,E,F,G, g_nom, g_denom

        # Solve Beta parameter
        A = (a[:,:,0]*r[:,:,1]) - (a[:,:,1]*r[:,:,0])
        B = (gam*a[:,:,1]*c[:,:,0]) - (gam * a[:,:,0] * c[:,:,1])
        b_nom =  A+ B
        b_denom = (a[:,:,0]*b[:,:,1]) - (a[:,:,1]*b[:,:,0])
        zeros = (b_denom == 0.).nonzero(as_tuple=True)
        b_denom[zeros[0], zeros[1]] = e
        beta = b_nom/b_denom
        

        del A,B,b_nom,b_denom

        a_nom = r[:,:,0]-((beta*b[:,:,0]) + (gam*c[:,:,0]))
        a_denom = a[:,:,0]
        zeros = (a_denom == 0.).nonzero(as_tuple=True)
        a_denom[zeros[0], zeros[1]] = e
        alpha = a_nom/ a_denom

        del a_nom,a_denom, zeros, a, b, c

        # Define mask to fetch indices where intersections occur
        beta_mask = torch.gt(beta, 0.)
        gam_mask = torch.gt(gam, 0.)
        alpha_mask = torch.gt(alpha, 0.)
        masks = beta_mask*alpha_mask*gam_mask 
        masks = masks.unsqueeze(2)
        on_surf_indices = (masks == True).nonzero(as_tuple=True)
        
        on_surf_indices = torch.cat([on_surf_indices[0].unsqueeze(1), on_surf_indices[1].unsqueeze(1)], dim=1)

        return on_surf_indices
    
    def get_intersection(self, indices, r, t):
        """Get intersection of xyz, t and indices (ref to plane-ray matrix) form

            Return:
             ray_indices, dist_indices, Tensors, matrices containing only valid r and t values
        """
        # tranfer intersections where our desired indices point
        ray_indices = torch.full_like(r, self.max_d).to(self.device)
        ray_indices[indices[:,0], indices[:,1]]  = r[indices[:,0], indices[:,1]]
        
        # transfer t-intersections where our desired indices point
        ts = t.unsqueeze(2)
        dist_indices = torch.full_like(ts, self.max_d).to(self.device)
        dist_indices[indices[:,0], indices[:,1]]  = ts[indices[:,0], indices[:,1]]

        return ray_indices, dist_indices

    def get_tn_tf(self, t_, indices):
        """Returns dictionary of viable ray-surface intersections with tn,tf
            Args:
                t_, Tensor, batch of intersection points and index pointer to global matrices

            Return:
                tn, tf, Tensor, matrices containing the near and far collision points relative to angle of incident

            Notes:
                We use good ol trigonometry to determine the thicnkess of a surface projected along our ray given the direction of the face (normals)
                and ray, and a known surface thicnkess of delta along the face normal. We then take the t-scalar for each point at each end
                of the projection and define these as [tn, tf] where tn < tf.
                We return this as a dict where ray indices and surface indeces can be used to read these values.s
        """
        delta = .001 # 5cm

        # Initialise Dict
        torch.cuda.empty_cache()

        tn_matrix = torch.full_like(t_, self.max_d).to(self.device)
        tf_matrix = torch.full_like(t_, self.max_d+1.).to(self.device)
        normals = self.surface_normals[indices[:,0]].unsqueeze(1) # .unsqueeze(1).repeat(1,t_.shape[1],1)
        hypotenuse = self.d[indices[:,1]].unsqueeze(2) # .unsqueeze(0).repeat(t_.shape[0],1,1)
        
        n_dot_h = torch.bmm(normals, hypotenuse).squeeze(1).squeeze(1)
        normals = self.surface_normals[indices[:,0]]
        hypotenuse = self.d[indices[:,1]]

        hypotensuse_mod = hypotenuse.pow(2).sum(dim=1).pow(0.5)
        norm_mod = normals.pow(2).sum(dim=1).pow(0.5)

        h_x_n = delta * hypotensuse_mod * norm_mod

        pd = (h_x_n / n_dot_h).unsqueeze(1)
        t1 = t_[indices[:,0],indices[:,1]]
        t2 = t1 + pd
        
        tntf = torch.sort(torch.cat([t1,t2], dim=1).unsqueeze(2), dim=1).values

        tn_matrix[indices[:,0],indices[:,1]] = tntf[:, 0]
        tf_matrix[indices[:,0],indices[:,1]] = tntf[:, 1]        
        
        return tn_matrix, tf_matrix

    def main(self, 
        surface_points,
        surface_normals,
        o,
        d,
        ):
        self.surface_bounds = surface_points.to(self.device)
        self.surface_normals = surface_normals.to(self.device)

        self.o = o
        self.d = d

        t = self.get_ray_plane_intersections_t()
        r = self.tansform_t_to_xyz(t)
        on_surface_indices = self.on_surface_filter(r)
        r_, t_ = self.get_intersection(on_surface_indices, r, t)
        tn_matrix, tf_matrix = self.get_tn_tf(t_, on_surface_indices)

        return r_, t_, tn_matrix, tf_matrix, on_surface_indices
