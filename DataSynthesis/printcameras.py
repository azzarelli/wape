# COLLAPSED
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import plotly_utils as vis
import plotly.graph_objects as go
webdocs_layout = go.Layout(
    scene=dict(
        aspectmode="data",
        xaxis=dict(showspikes=False),
        yaxis=dict(showspikes=False),
        zaxis=dict(showspikes=False),
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
    ),
    scene_camera=dict(up=dict(x=0, y=1, z=0)),
    margin=dict(r=0, b=10, l=0, t=10),
    hovermode=False,
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
)

cx = 2.0
cy = 2.0
fx = 10.0
fy = 10.0

num_samples = 3
near_plane = 1
far_plane = 3

c2w = torch.eye(4)[None, :3, :]
camera = Cameras(fx=fx, fy=fy, cx=cx, cy=cy, camera_to_worlds=c2w, camera_type=CameraType.PERSPECTIVE)
ray_bundle = camera.generate_rays(camera_indices=0)

bins = torch.linspace(near_plane, far_plane, num_samples + 1)[..., None]
ray_samples = ray_bundle.get_ray_samples(bin_starts=bins[:-1, :], bin_ends=bins[1:, :])

vis_rays = vis.get_ray_bundle_lines(ray_bundle, color="teal", length=far_plane)


fig = go.Figure(data=[vis_rays] + vis.get_frustums_mesh_list(ray_samples.frustums), layout=webdocs_layout)
# fig.show()