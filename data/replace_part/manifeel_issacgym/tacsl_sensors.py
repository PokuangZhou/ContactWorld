'''
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Tactile Sensing Modules
----------------
Implementation of visuo-tactile sensing module to generate tactile RGB image or tactile force field.
'''

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import tf_combine, tf_inverse, get_euler_xyz
from isaacgym import torch_utils as tu

from isaacgymenvs.tacsl_sensors.tactile_utils.gelsight_render import gelsightRender

import os
import itertools
import numpy as np
import torch
import trimesh
from urdfpy import URDF
import yaml
# import time
from collections import defaultdict
import pickle

from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from collections.abc import Iterable

# import matplotlib.pyplot as plt

import math
# optional
try:
    from pytorch3d.ops import sample_farthest_points
    _HAS_PYTORCH3D = True
except Exception:
    _HAS_PYTORCH3D = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOOKUP_TABLE_PATH = os.path.join(CURRENT_DIR, "my_vt_lookup_table_impedance.pkl")

def get_camera_config(sensor_type, tactile_camera_name, actor_name, attach_link_name):
    """
    Get the camera configuration based on the sensor type.

    Args:
        sensor_type (str): Type of the sensor (e.g., 'gelsight_r15', 'gs_mini').
        tactile_camera_name (str): Name of the tactile camera.
        actor_name (str): Name of the actor on which the tactile sensor is attached.
        attach_link_name (str): Name of the link to which the camera is attached.

    Returns:
        dict: Tactile camera configuration.
    """
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if sensor_type == 'gelsight_r15':
        config_filepath = os.path.join(parent_dir, 'configs', 'gelsight_r15.yaml')
    elif sensor_type == 'gs_mini':
        config_filepath = os.path.join(parent_dir, 'configs', 'gs_mini.yaml')
    else:
        raise NotImplementedError(f'Sensor type {sensor_type} not recognized')

    with open(config_filepath) as stream:
        tactile_camera_config = yaml.safe_load(stream)
    tip_to_cam_pos = [0.0, 0.0, -tactile_camera_config['camera_dist']]
    tip_to_cam_quat = R.from_euler('xyz', tactile_camera_config['tip_to_cam_euler']).as_quat()
    tactile_camera_config['camera_pose'] = [tip_to_cam_pos, tip_to_cam_quat.tolist()]
    tactile_camera_config['image_size'] = [tactile_camera_config['height'], tactile_camera_config['width']]

    fx, fy = tactile_camera_config['camera_dist'], tactile_camera_config['camera_dist']
    fov_x = 2 * np.arctan(tactile_camera_config['cx'] / fx) * 180 / np.pi
    fov_y = 2 * np.arctan(tactile_camera_config['cy'] / fy) * 180 / np.pi
    tactile_camera_config['horizontal_fov'] = float(fov_x)

    tactile_camera_config['name'] = tactile_camera_name
    tactile_camera_config['actor_name'] = actor_name
    tactile_camera_config['attach_link_name'] = attach_link_name
    return tactile_camera_config


class TactileBase:
    """
    Base class for tactile sensors.
    Provides methods for configuring compliant dynamics.
    """
    def configure_compliant_dynamics(self, actor_handle, elastomer_link_name,
                                     compliance_stiffness, compliant_damping, use_acceleration_spring=False):
        """
        Configure the compliant dynamics for a given actor and link.

        Args:
            actor_handle: Handle for the actor.
            elastomer_link_name (str): Name of the elastomer link.
            compliance_stiffness (float or list): Compliance stiffness value(s).
            compliant_damping (float or list): Compliant damping value(s).
            use_acceleration_spring (bool): Whether to use acceleration spring.
        """
        if not isinstance(compliance_stiffness, Iterable):
            compliance_stiffness = [compliance_stiffness] * len(self.env_ptrs)
        if not isinstance(compliant_damping, Iterable):
            compliant_damping = [compliant_damping] * len(self.env_ptrs)

        for env_id, env_ptr in enumerate(self.env_ptrs):
            body_name_idx_map = self.gym.get_actor_rigid_body_dict(env_ptr, actor_handle)
            body_names = self.gym.get_actor_rigid_body_names(env_ptr, actor_handle)
            elastomer_body_id = body_name_idx_map[elastomer_link_name]
            assert body_names[elastomer_body_id] == elastomer_link_name, 'order of rigid body does not agree'
            rs_props = self.gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)
            rb_shape_indices = self.gym.get_actor_rigid_body_shape_indices(env_ptr, actor_handle)
            elastomer_shape_id = rb_shape_indices[elastomer_body_id].start

            rs_props[elastomer_shape_id].compliance = compliance_stiffness[env_id]
            rs_props[elastomer_shape_id].compliant_damping = compliant_damping[env_id]
            rs_props[elastomer_shape_id].use_acceleration_spring = use_acceleration_spring
            self.gym.set_actor_rigid_shape_properties(env_ptr, actor_handle, rs_props)

    def set_compliant_dynamics_for_tactile_sensors(self, tactile_sensor_configs):
        """
        Set the compliant dynamics for multiple tactile sensors.

        Args:
            tactile_sensor_configs (list): List of tactile sensor configurations.
        """
        for tactile_sensor_config in tactile_sensor_configs:
            self.configure_compliant_dynamics(actor_handle=tactile_sensor_config['actor_handle'],
                                              elastomer_link_name=tactile_sensor_config['elastomer_link_name'],
                                              compliance_stiffness=tactile_sensor_config['compliance_stiffness'],
                                              compliant_damping=tactile_sensor_config['compliant_damping'],
                                              use_acceleration_spring=tactile_sensor_config['use_acceleration_spring'])


class CameraSensor:
    """
    Class for managing camera sensors.
    Provides methods for creating and managing camera actors and tensors.
    """
    def create_camera_actors(self, camera_spec_dict):
        """
        Create camera actors based on the camera specification dictionary.
        # Note: This should be called once, as IsaacGym's global camera indexing expects all cameras of env 0 be created before env 1, and so on.

        Args:
            camera_spec_dict (dict): Dictionary of camera specifications.

        Returns:
            tuple: List of camera handles and list of camera tensors.
        """
        camera_handles_list = []
        camera_tensors_list = []

        for i in range(self.num_envs):
            env_ptr = self.env_ptrs[i]
            env_camera_handles = self.setup_env_cameras(env_ptr, camera_spec_dict)
            camera_handles_list.append(env_camera_handles)

            env_camera_tensors = self.create_tensors_for_env_cameras(env_ptr, env_camera_handles, camera_spec_dict)
            camera_tensors_list.append(env_camera_tensors)
        return camera_handles_list, camera_tensors_list

    def create_tensors_for_env_cameras(self, env_ptr, env_camera_handles, camera_spec_dict):
        """
        Create tensors for environment cameras.

        Args:
            env_ptr: Pointer to the environment.
            env_camera_handles (dict): Dictionary of camera handles.
            camera_spec_dict (dict): Dictionary of camera specifications.

        Returns:
            dict: Dictionary of environment camera tensors.
        """
        env_camera_tensors = {}
        for name in env_camera_handles:
            camera_handle = env_camera_handles[name]
            if camera_spec_dict[name].image_type == 'rgb':
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                     gymapi.IMAGE_COLOR)
            elif camera_spec_dict[name].image_type == 'depth':
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                     gymapi.IMAGE_DEPTH)
            else:
                raise NotImplementedError(f'Camera type {camera_spec_dict[name].image_type} not supported')

            # wrap camera tensor in a pytorch tensor
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)

            # store references to the tensor that gets updated when render_all_camera_sensors
            env_camera_tensors[name] = torch_camera_tensor
        return env_camera_tensors

    def setup_env_cameras(self, env_ptr, camera_spec_dict):
        """
        Set up environment cameras.

        Args:
            env_ptr: Pointer to the environment.
            camera_spec_dict (dict): Dictionary of camera specifications.

        Returns:
            dict: Dictionary of camera handles.
        """
        camera_handles = {}
        for name, camera_spec in camera_spec_dict.items():
            camera_properties = gymapi.CameraProperties()
            camera_properties.height = camera_spec.image_size[0]
            camera_properties.width = camera_spec.image_size[1]
            camera_properties.enable_tensors = True
            camera_properties.horizontal_fov = camera_spec.horizontal_fov
            if 'near_plane' in camera_spec:
                camera_properties.near_plane = camera_spec.near_plane

            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_properties)
            camera_handles[name] = camera_handle

            if camera_spec.is_body_camera:
                actor_handle = self.gym.find_actor_handle(env_ptr, camera_spec.actor_name)
                robot_body_handle = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle,
                                                                          camera_spec.attach_link_name)

                self.gym.attach_camera_to_body(
                    camera_handle,
                    env_ptr,
                    robot_body_handle,
                    gymapi.Transform(gymapi.Vec3(*camera_spec.camera_pose[0]),
                                     gymapi.Quat(*camera_spec.camera_pose[1])),
                    gymapi.FOLLOW_TRANSFORM,
                )
            else:
                transform = gymapi.Transform(gymapi.Vec3(*camera_spec.camera_pose[0]),
                                             gymapi.Quat(*camera_spec.camera_pose[1]))
                self.gym.set_camera_transform(camera_handle, env_ptr, transform)
        return camera_handles

    def get_camera_image_tensors_dict(self):
        """
        Get the dictionary of camera image tensors.

        Returns:
            dict: Dictionary of camera image tensors.
        """
        # transforms and information must be communicated from the physics simulation into the graphics system
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        camera_image_tensors_dict = dict()

        for name in self.camera_spec_dict:
            camera_spec = self.camera_spec_dict[name]
            if camera_spec['image_type'] == 'rgb':
                num_channels = 3
                camera_images = torch.zeros(
                    (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1], num_channels),
                    device=self.device, dtype=torch.uint8)
                for id in np.arange(self.num_envs):
                    camera_images[id] = self.camera_tensors_list[id][name][:, :, :num_channels].clone()
            elif camera_spec['image_type'] == 'depth':
                num_channels = 1
                camera_images = torch.zeros(
                    (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1]),
                    device=self.device, dtype=torch.float)
                for id in np.arange(self.num_envs):
                    # Note that isaac gym returns negative depth
                    # See: https://carbon-gym.gitlab-master-pages.nvidia.com/carbgym/graphics.html?highlight=image_depth#camera-image-types
                    camera_images[id] = self.camera_tensors_list[id][name][:, :].clone() * -1.
                    camera_images[id][camera_images[id] == np.inf] = 0.0
            else:
                raise NotImplementedError(f'Image type {camera_spec["image_type"]} not supported!')
            camera_image_tensors_dict[name] = camera_images

        return camera_image_tensors_dict


    def _depth_to_pointcloud_world_from_camera(
        self,
        env_id,
        camera_name,
        depth,
        min_depth=1e-4,
        max_depth=None,
    ):
        """
        Convert one camera depth image to world-frame point cloud using
        Isaac Gym camera view/projection matrices.

        Args:
            env_id: int
            camera_name: str
            depth: [H, W], positive depth from get_camera_image_tensors_dict()

        Returns:
            pts_world: [N, 3]
            valid_mask_flat: [H*W]
        """
        env_ptr = self.env_ptrs[env_id]
        cam_handle = self.camera_handles_list[env_id][camera_name]

        H, W = depth.shape
        device = depth.device
        dtype = depth.dtype

        view = torch.tensor(
            self.gym.get_camera_view_matrix(self.sim, env_ptr, cam_handle),
            device=device,
            dtype=dtype,
        )  # [4, 4]

        proj = torch.tensor(
            self.gym.get_camera_proj_matrix(self.sim, env_ptr, cam_handle),
            device=device,
            dtype=dtype,
        )  # [4, 4]

        vinv = torch.inverse(view)

        fu = 2.0 / proj[0, 0]
        fv = 2.0 / proj[1, 1]

        center_u = W / 2.0
        center_v = H / 2.0

        u = torch.arange(W, device=device, dtype=dtype)
        v = torch.arange(H, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        Z = depth
        valid = Z > min_depth
        if max_depth is not None:
            valid = valid & (Z < max_depth)

        X = ((uu - center_u) / W) * Z * fu
        Y = ((vv - center_v) / H) * Z * fv

        ones = torch.ones_like(Z)
        pts_cam_h = torch.stack([X, Y, Z, ones], dim=-1).reshape(-1, 4)  # [HW, 4]

        valid_flat = valid.reshape(-1)
        pts_cam_h = pts_cam_h[valid_flat]

        pts_world_h = pts_cam_h @ vinv.T
        pts_world = pts_world_h[:, :3]

        return pts_world, valid_flat


    def _transform_points_world_to_base(self, pc_world, base_pos, base_quat_xyzw):
        """
        Args:
            pc_world: [B, N, 3]
            base_pos: [B, 3]
            base_quat_xyzw: [B, 4]
        Returns:
            pc_base: [B, N, 3]
        """
        base_quat_inv, base_pos_inv = tf_inverse(base_quat_xyzw, base_pos)

        pc_shift = pc_world - base_pos.unsqueeze(1)
        quat_expand = base_quat_inv.unsqueeze(1).expand(-1, pc_world.shape[1], -1)
        pc_base = tu.quat_apply(quat_expand.reshape(-1, 4), pc_shift.reshape(-1, 3))
        pc_base = pc_base.reshape_as(pc_world)
        return pc_base


    def _downsample_pointcloud_fixed_count(self, pts, num_points):
        """
        Args:
            pts: [N, C], where C can be 3 or 6
        Returns:
            pts_ds: [num_points, C]
        """
        device = pts.device
        N, C = pts.shape

        if N == 0:
            return torch.zeros((num_points, C), device=device, dtype=torch.float32)

        if N == num_points:
            return pts

        if N < num_points:
            repeat_idx = torch.randint(0, N, (num_points - N,), device=device)
            return torch.cat([pts, pts[repeat_idx]], dim=0)

        if "_HAS_PYTORCH3D" in globals() and _HAS_PYTORCH3D:
            xyz = pts[:, :3].unsqueeze(0)
            _, idx = sample_farthest_points(xyz, K=num_points)
            idx = idx[0]
            return pts[idx]

        idx = torch.randperm(N, device=device)[:num_points]
        return pts[idx]


    def get_merged_pointcloud_base(
        self,
        camera_names=None,
        rgb_camera_map=None,
        num_points=1024,
        crop_bounds=None,
        min_depth=1e-4,
        max_depth=None,
        include_rgb=True,
    ):
        """
        Build merged multi-view point cloud in BASE frame.

        Args:
            camera_names: list[str], e.g. ["front_depth", "wrist_depth"]
            rgb_camera_map: dict, e.g. {"front_depth": "front", "wrist_depth": "wrist"}
            num_points: output point count
            crop_bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) in BASE frame
            include_rgb: True -> xyzrgb, False -> xyz

        Returns:
            [B, num_points, 3 or 6]
        """
        assert hasattr(self, "franka_base_pos"), "Need franka_base_pos."
        assert hasattr(self, "franka_base_quat"), "Need franka_base_quat."

        image_dict = self.get_camera_image_tensors_dict()

        if camera_names is None:
            camera_names = [
                name for name, spec in self.camera_spec_dict.items()
                if spec["image_type"] == "depth"
            ]

        if len(camera_names) == 0:
            raise RuntimeError("No depth cameras found for point cloud generation.")

        if include_rgb and rgb_camera_map is None:
            raise ValueError("include_rgb=True requires rgb_camera_map.")

        feat_dim = 6 if include_rgb else 3
        merged_world = []

        for env_id in range(self.num_envs):
            pts_env = []

            for cam_name in camera_names:
                depth = image_dict[cam_name][env_id].float()  # [H, W]

                pts_world_xyz, valid_flat = self._depth_to_pointcloud_world_from_camera(
                    env_id=env_id,
                    camera_name=cam_name,
                    depth=depth,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )

                if include_rgb:
                    rgb_name = rgb_camera_map[cam_name]
                    rgb = image_dict[rgb_name][env_id]

                    if rgb.dtype == torch.uint8:
                        rgb = rgb.float() / 255.0
                    else:
                        rgb = rgb.float()

                    rgb_flat = rgb.reshape(-1, 3)
                    pts_rgb = rgb_flat[valid_flat]
                    pts_world = torch.cat([pts_world_xyz, pts_rgb], dim=-1)  # [N, 6]
                else:
                    pts_world = pts_world_xyz

                pts_env.append(pts_world)

            if len(pts_env) > 0:
                merged_world.append(torch.cat(pts_env, dim=0))
            else:
                merged_world.append(torch.zeros((0, feat_dim), device=self.device, dtype=torch.float32))

        # world -> base
        merged_base = []
        for b in range(self.num_envs):
            pts = merged_world[b]
            if pts.shape[0] == 0:
                merged_base.append(pts)
                continue

            xyz_world = pts[:, :3].unsqueeze(0)  # [1, N, 3]
            xyz_base = self._transform_points_world_to_base(
                xyz_world,
                self.franka_base_pos[b:b+1],
                self.franka_base_quat[b:b+1],
            )[0]

            if include_rgb:
                rgb = pts[:, 3:]
                pts_base = torch.cat([xyz_base, rgb], dim=-1)
            else:
                pts_base = xyz_base

            merged_base.append(pts_base)

        # crop in BASE frame
        cropped_base = []
        if crop_bounds is None:
            cropped_base = merged_base
        else:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = crop_bounds
            for pts in merged_base:
                if pts.shape[0] == 0:
                    cropped_base.append(pts)
                    continue
                xyz = pts[:, :3]
                mask = (
                    (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
                    (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
                    (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
                )
                cropped_base.append(pts[mask])

        out = []
        for pts in cropped_base:
            out.append(self._downsample_pointcloud_fixed_count(pts, num_points))

        return torch.stack(out, dim=0)


class TactileRGBSensor(TactileBase, CameraSensor):
    """
    Class for simulating tactile RGB sensors.
    Inherits from TactileBase and CameraSensor.
    """
    def __init__(self):
        super().__init__()
        self.taxim_gelsight = None
        self.has_tactile_rgb = False
        self.nominal_tactile = None

    def get_tactile_rgb_camera_configs(self, tactile_sensor_configs):
        """
        Get the tactile RGB camera configurations.

        Args:
            tactile_sensor_configs (list): List of tactile sensor configurations.

        Returns:
            dict: Dictionary of tactile camera specifications.
        """
        camera_spec_dict = dict()

        for tactile_sensor_config in tactile_sensor_configs:
            camera_config = get_camera_config(tactile_sensor_config['sensor_type'],
                                              tactile_sensor_config['tactile_camera_name'],
                                              tactile_sensor_config['actor_name'],
                                              tactile_sensor_config['attach_link_name'])
            camera_config = OmegaConf.create(camera_config)
            tactile_camera_spec_dict = {camera_config['name']: camera_config}
            camera_spec_dict.update(tactile_camera_spec_dict)

        self.taxim_gelsight = gelsightRender(tactile_sensor_configs[0]['sensor_type'], device=self.device)
        self.has_tactile_rgb = True

        return camera_spec_dict

    def initialize_tactile_rgb_camera(self):
        """
        Initialize the tactile RGB camera by capturing a nominal tactile image with no indentation on the elastomer.
        """
        image_dict = self.get_camera_image_tensors_dict()
        self.nominal_tactile = {k: image_dict[k][:1] for k in image_dict.keys() if 'tactile' in k}

    def visualize_tactile_camera_frame(self, tip_to_cam_pos, tip_to_cam_quat):
        gelsight_handle = self.actor_handles['gelsight_finger']
        elastomer_tip_link_handle = self.gym.find_actor_rigid_body_handle(self.env_ptrs[0], gelsight_handle,
                                                                          'elastomer_tip')
        tip_to_cam_pos_tensor = torch.tensor(tip_to_cam_pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        tip_to_cam_quat_tensor = torch.tensor(tip_to_cam_quat, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        world_to_camera_trans_new = tf_combine(self.body_quat[:, elastomer_tip_link_handle],
                                               self.body_pos[:, elastomer_tip_link_handle],
                                               tip_to_cam_quat_tensor,
                                               tip_to_cam_pos_tensor)
        for env_id in range(len(self.env_ptrs)):
            self.draw_frame_in_env(env_id, (world_to_camera_trans_new[1][env_id],
                                            world_to_camera_trans_new[0][env_id]))

    def get_camera_image_tensors_dict(self):
        """
        Get the dictionary of camera image tensors, including tactile RGB images.

        Returns:
            dict: Dictionary of camera image tensors.
        """
        camera_image_tensors_dict = super().get_camera_image_tensors_dict()

        # Compute tactile RGB from tactile depth
        if hasattr(self, 'has_tactile_rgb') and self.nominal_tactile:
            for k in self.nominal_tactile:
                depth_image = self.nominal_tactile[k] - camera_image_tensors_dict[k]  # [B, H, W, 3] or [B, H, W, C]
                taxim_render_all = self.taxim_gelsight.render_tensorized(depth_image)

                # convert tactile depth to single-channel if needed
                if depth_image.ndim == 4 and depth_image.shape[-1] > 1:
                    depth_scalar = depth_image[..., 0]
                else:
                    depth_scalar = depth_image.squeeze(-1) if depth_image.ndim == 4 else depth_image
                
                depth_scalar = torch.clamp(depth_scalar, min=0)
                depth_scalar = depth_scalar / (depth_scalar.max() + 1e-6)
                
                # map to your desired names
                if k == 'right_tactile_camera':
                    camera_image_tensors_dict['tactile_depth_right'] = depth_scalar
                    camera_image_tensors_dict['tactile_rgb_right'] = taxim_render_all

        return camera_image_tensors_dict




# Initialize plot-related buffers
vt_world_buffer = []
plot_counter = 0
PLOT_INTERVAL = 10
BUFFER_SIZE = 100

class TactileFieldSensor(TactileBase):
    """
    Class for simulating tactile field sensors.
    Inherits from TactileBase.
    """
    def __init__(self):
        super().__init__()
        self.tactile_pos_local, self.tactile_quat_local = None, None
        self.sdf_tool = None
        self.tactile_shear_field_configs_dict = None
        self.sdf, self.indenter_mesh, self.indenter_mesh_local_tf = None, None, None

        # initialize coefficients
        self.tactile_kn = 1.
        self.tactile_damping = 0.003
        self.tactile_mu = 2. # 2
        self.tactile_kt = 0.1 # 0.5
        
    def setup_tactile_force_field(self, sdf_tool, num_tactile_rows, num_tactile_cols, tactile_shear_field_configs):
        """
        Set up the tactile force field sensing.

        Args:
            sdf_tool (str): Tool for signed distance field (SDF) calculation.
            num_tactile_rows (int): Number of rows of tactile points.
            num_tactile_cols (int): Number of columns of tactile points.
            tactile_shear_field_configs (dict): Configuration for the tactile shear field.

        Returns:
            tensor: SDF tensor.
        """
        self.sdf_tool = sdf_tool
        self.tactile_shear_field_configs_dict = self.post_process_shear_field_configs(tactile_shear_field_configs)
        # Set up tactile sdf tensors
        sdf_tensor = None
        if sdf_tool == 'physx':
            # This should be called before calling self.gym.prepare_sim(self.sim)
            num_queries_per_env = num_tactile_rows * num_tactile_cols
            _sdf_tensor = self.gym.acquire_sdf_view_tensor(self.sim, 1, num_queries_per_env)
            sdf_tensor = gymtorch.wrap_tensor(_sdf_tensor)
        return sdf_tensor

    def get_elastomer_to_tip_transform(self, actor_name, elastomer_link_name, elastomer_tip_link_name):
        """
        Get the transformation from the elastomer to the tip.
        Note: This is the same for a given sensor_type so should probably go into a config file

        Args:
            actor_name (str): Name of the actor.
            elastomer_link_name (str): Name of the elastomer link.
            elastomer_tip_link_name (str): Name of the elastomer tip link.

        Returns:
            tuple: Quaternion and position of the transformation.
        """
        elastomer_parent_actor_handle = self.actor_handles[actor_name]
        elastomer_link_handle = self.gym.find_actor_rigid_body_handle(self.env_ptrs[0], elastomer_parent_actor_handle, elastomer_link_name)
        elastomer_tip_link_handle = self.gym.find_actor_rigid_body_handle(self.env_ptrs[0], elastomer_parent_actor_handle,
                                                                          elastomer_tip_link_name)
        assert elastomer_link_handle > -1, 'elastomer_link_handle does not exist'
        assert elastomer_tip_link_handle > -1, 'elastomer_tip_link_handle does not exist'
        elastomer_to_tip_link_trans = tf_combine(*tf_inverse(self.body_quat[:, elastomer_link_handle],
                                                             self.body_pos[:, elastomer_link_handle]),
                                                 self.body_quat[:, elastomer_tip_link_handle],
                                                 self.body_pos[:, elastomer_tip_link_handle])
        elastomer_to_tip_link_quat, elastomer_to_tip_link_pos = elastomer_to_tip_link_trans
        return elastomer_to_tip_link_quat, elastomer_to_tip_link_pos

    def generate_tactile_points(self, elastomer_parent_urdf_path, elastomer_link_name, elastomer_tip_link_name,
                                elastomer_actor_name, num_divs=[20, 25], margin=0.003, visualize=False):
        """
        Generate tactile points on the elastomer.

        Args:
            elastomer_parent_urdf_path (str): Path to the elastomer parent URDF.
            elastomer_link_name (str): Name of the elastomer link.
            elastomer_tip_link_name (str): Name of the elastomer tip link.
            elastomer_actor_name (str): Name of the elastomer actor.
            num_divs (list): Number of divisions for the tactile points.
            margin (float): Margin for the tactile points.
            visualize (bool): Whether to visualize the points.

        Returns:
            tuple: Tactile points positions and quaternions.
        """
        robot = URDF.load(elastomer_parent_urdf_path)
        mesh = robot.link_map[elastomer_link_name].visuals[0].geometry.mesh.meshes[0]

        # generate grid on elastomer
        elastomer_dims = np.diff(mesh.bounds, axis=0).squeeze()
        slim_axis = np.argmin(elastomer_dims)   # determine flat axis of elastomer
        _, elastomer_to_tip_link_pos = self.get_elastomer_to_tip_transform(
            actor_name=elastomer_actor_name,
            elastomer_link_name=elastomer_link_name,
            elastomer_tip_link_name=elastomer_tip_link_name)

        # determine gap between adjacent tactile points
        axis_idxs = list(range(3))
        axis_idxs.remove(slim_axis)     # remove slim idx
        div_sz = (elastomer_dims[axis_idxs] - margin * 2.) / (np.array(num_divs) + 1)
        tactile_points_dx = min(div_sz)

        # sample points on the flat plane
        planar_grid_points = []
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.
        idx = 0
        for axis_i in range(3):
            if axis_i == slim_axis:
                # On the slim axis, place a point far away so ray is pointing at the elastomer tip
                planar_grid_points.append([np.sign(elastomer_to_tip_link_pos[0][slim_axis].item())])
            else:
                axis_grid_points = np.linspace(center[axis_i] - tactile_points_dx * (num_divs[idx] + 1.) / 2., center[axis_i] + tactile_points_dx * (num_divs[idx] + 1.) / 2., num_divs[idx] + 2)
                planar_grid_points.append(axis_grid_points[1:-1])  # leave out the extreme corners
                idx += 1

        grid_corners = itertools.product(planar_grid_points[0], planar_grid_points[1], planar_grid_points[2])
        grid_corners = np.array(list(grid_corners))

        # project ray in positive y direction on the mesh # NOTE [Jie]: number of points is wrong when num_divs = 40
        mesh_data = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        ray_dir = np.array([0, 0, 0])
        ray_dir[slim_axis] = -np.sign(elastomer_to_tip_link_pos[0][slim_axis].item())   # ray point towards elastomer
        index_tri, index_ray, locations = mesh_data.intersects_id(grid_corners,
                                                                  np.tile([ray_dir], (grid_corners.shape[0], 1)),
                                                                  return_locations=True, multiple_hits=False)

        if visualize:
            query_pointcloud = trimesh.PointCloud(locations, colors=(0., 0., 1.))

            trimesh.Scene([mesh, query_pointcloud]).show()

        if len(index_ray) != len(grid_corners):
            raise ValueError("Fewer number of tactile points")

        tactile_points = locations[index_ray.argsort()]
        tactile_points_pos_tensor = torch.tensor(tactile_points, dtype=torch.float32, device=self.device)
        rotation = (0, 0, -np.pi) # NOTE [Jie]: assume tactile frame rotation are all the same
        rotation = R.from_euler('xyz', rotation).as_quat()
        tactile_points_quat_tensor = torch.tensor(rotation, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(len(tactile_points), 1)
        return tactile_points_pos_tensor, tactile_points_quat_tensor

    def get_tactile_points_in_world(self, tactile_points_pos_local, tactile_points_quat_local, elastomer_link_id):
        """
        Get the tactile points in the world frame.

        Args:
            tactile_points_pos_local (tensor): Local positions of the tactile points.
            tactile_points_quat_local (tensor): Local quaternions of the tactile points.
            elastomer_link_id (int): ID of the elastomer link.

        Returns:
            tuple: Positions and quaternions of the tactile points in the world frame.
        """
        num_tactile_points = tactile_points_pos_local.shape[0]
        body_pos = self.body_pos[:, elastomer_link_id].unsqueeze(1).expand(self.num_envs, num_tactile_points, 3)
        body_quat = self.body_quat[:, elastomer_link_id].unsqueeze(1).expand(self.num_envs, num_tactile_points, 4)
        tactile_points_pos_tmp = tactile_points_pos_local.unsqueeze(0).expand(self.num_envs, num_tactile_points, 3)
        tactile_points_quat_tmp = tactile_points_quat_local.unsqueeze(0).expand(self.num_envs, num_tactile_points, 4)
        quat_list_per_env, pos_list_per_env = tf_combine(body_quat, body_pos,
                                                         tactile_points_quat_tmp, tactile_points_pos_tmp)

        return pos_list_per_env, quat_list_per_env

    ### Trimesh
    def construct_sdf_with_trimesh(self, mesh):
        """
        Construct the signed distance field (SDF) using trimesh.

        Args:
            mesh (trimesh.Trimesh): Trimesh object.

        Returns:
            trimesh.proximity.ProximityQuery: Proximity query object.
        """
        proximity_query = trimesh.proximity.ProximityQuery(mesh)
        return proximity_query

    def query_distance_and_normal_with_trimesh(self, sdf, query_points, visualize=False):
        """
        Query distance and normal using trimesh.

        Args:
            sdf (trimesh.proximity.ProximityQuery): Proximity query object.
            query_points (tensor): Points to query.
            visualize (bool): Whether to visualize the results.

        Returns:
            tuple: Mask, distance, and normal vectors.
        """
        dtype = query_points.dtype
        query_points_np = query_points.cpu().numpy()

        distance_np = sdf.signed_distance(query_points_np)
        mask = (distance_np > 0.) # positive for penetration

        distance = torch.tensor(distance_np, dtype = dtype, device = self.device)
        normal = torch.zeros(query_points.shape, dtype = dtype, device = self.device)
        if mask.sum() > 0:
            closest_np, _, _ = sdf.on_surface(query_points_np[mask])

            closest = torch.tensor(closest_np, dtype = dtype, device = self.device)

            normal[mask] = tu.normalize(closest - query_points[mask])

            if visualize:
                pointcloud_other_query = trimesh.PointCloud(query_points_np[~mask], colors=(0., 0., 1.))
                pointcloud_query = trimesh.PointCloud(query_points_np[mask], colors=(0., 1., 0.))
                pointcloud_closest = trimesh.PointCloud(closest_np, colors=(1., 0., 0.))

                trimesh.Scene([self.indenter_mesh, pointcloud_other_query, pointcloud_query, pointcloud_closest]).show()

        return mask, distance, normal

    ###  pysdf
    def construct_sdf_with_pysdf(self, mesh):
        """
        Construct the signed distance field (SDF) using pysdf.

        Args:
            mesh (trimesh.Trimesh): Trimesh object.

        Returns:
            SDF: SDF object.
        """
        from pysdf import SDF
        sdf = SDF(mesh.vertices, mesh.faces)
        return sdf

    def query_distance_and_normal_with_pysdf(self, sdf, query_points, visualize=False):
        """
        Query distance and normal using pysdf.

        Args:
            sdf (SDF): SDF object.
            query_points (tensor): Points to query.
            visualize (bool): Whether to visualize the results.

        Returns:
            tuple: Mask, distance, and normal vectors.
        """
        dtype = query_points.dtype
        query_points_np = query_points.cpu().numpy()

        distance_np = sdf(query_points_np)
        mask = (distance_np > 0.) # positive for penetration

        distance = torch.tensor(distance_np, dtype = dtype, device = self.device)
        normal = torch.zeros(query_points.shape, dtype = dtype, device = self.device)

        if mask.sum() > 0:
            eps = 1e-5
            grad_x = sdf(query_points_np[mask] + np.array([eps, 0., 0.])) - sdf(query_points_np[mask] - np.array([eps, 0., 0.]))
            grad_y = sdf(query_points_np[mask] + np.array([0., eps, 0.])) - sdf(query_points_np[mask] - np.array([0., eps, 0.]))
            grad_z = sdf(query_points_np[mask] + np.array([0., 0., eps])) - sdf(query_points_np[mask] - np.array([0., 0., eps]))
            grad = torch.tensor(np.stack((grad_x, grad_y, grad_z), axis = 1), dtype = dtype, device = self.device) # check dimension
            normal[mask] = -tu.normalize(grad)

            if visualize:
                closest_np = query_points_np[mask] + np.expand_dims(distance_np[mask], axis = -1) * normal[mask].cpu().numpy()
                pointcloud_other_query = trimesh.PointCloud(query_points_np[~mask], colors=(0., 0., 1.))
                pointcloud_query = trimesh.PointCloud(query_points_np[mask], colors=(0., 1., 0.))
                pointcloud_closest = trimesh.PointCloud(closest_np, colors=(1., 0., 0.))

                trimesh.Scene([self.indenter_mesh, pointcloud_other_query, pointcloud_query, pointcloud_closest]).show()

        return mask, distance, normal

    ### PhysX SDF
    def query_distance_and_normal_with_physx(self, query_points, visualize=False):
        """
        Query distance and normal using PhysX.

        Args:
            query_points (tensor): Points to query.
            visualize (bool): Whether to visualize the results.

        Returns:
            tuple: Mask, distance, and normal vectors.
        """
        self.gym.refresh_sdf_view_tensor(self.sim,
                                         gymtorch.unwrap_tensor(self.sdf_shape_global_ids_per_env),
                                         gymtorch.unwrap_tensor(query_points))
        distance = -self.sdf_tensor[:, ..., 3]
        normal = self.sdf_tensor[:, ..., :3] # TODO: check direction of normal
        mask = (distance > 0.) # positive for penetration

        if mask.sum() > 0 and visualize:
            viz_env_id = 0
            pointcloud_other_query = trimesh.PointCloud(query_points[viz_env_id][~mask[viz_env_id]].cpu().numpy(),
                                                        colors=(0., 0., 1.))
            pointcloud_query = trimesh.PointCloud(query_points[viz_env_id][mask[viz_env_id]].cpu().numpy(),
                                                  colors=(0., 1., 0.))

            trimesh.Scene([self.indenter_mesh, pointcloud_other_query, pointcloud_query]).show()
        return mask, distance, normal

    def load_sdf_oracle_of_indenter(self, indenter_urdf_path, indenter_actor_name, indenter_rb_name):
        """
        Load the SDF oracle of the indenter.

        Args:
            indenter_urdf_path (str): Path to the indenter URDF.
            indenter_actor_name (str): Name of the indenter actor.
            indenter_rb_name (str): Name of the indenter rigid body.

        Returns:
            tuple: SDF object, indenter mesh, and indenter mesh local transformation.
        """
        print(f"✅{indenter_urdf_path}-{indenter_actor_name}-{indenter_rb_name}")
        robot = URDF.load(indenter_urdf_path)
        indenter_mesh = robot.links[-1].visuals[0].geometry.geometry.meshes[0]
        origin = robot.links[-1].visuals[0].origin
        tf_mat = origin
        tf_pos = tf_mat[0:3, 3]
        tf_quat = R.from_matrix(tf_mat[0:3, 0:3]).as_quat()
        # indenter_mesh_local_tf = (torch.tensor([tf_quat], dtype=torch.float, device=self.device), torch.tensor([tf_pos], dtype=torch.float, device=self.device))
        indenter_mesh_local_tf = (torch.tensor(tf_quat, dtype=torch.float, device=self.device).unsqueeze(0), 
                                  torch.tensor(tf_pos, dtype=torch.float, device=self.device).unsqueeze(0))

        if self.sdf_tool == 'trimesh':
            sdf = self.construct_sdf_with_trimesh(indenter_mesh)
        elif self.sdf_tool == 'pysdf':
            sdf = self.construct_sdf_with_pysdf(indenter_mesh)
        elif self.sdf_tool == 'physx':
            # intialize the sdf object indices
            # self.sdf_shape_global_ids_per_env = torch.zeros((self.num_envs, 1), dtype=torch.int32, device=self.device)
            sdf_shape_global_ids_per_env = torch.zeros((self.num_envs, 1), dtype=torch.int32, device=self.device)
            indenter_actor_handle = self.actor_handles[indenter_actor_name]
            for env_id, env_ptr in enumerate(self.env_ptrs):
                indenter_rb_id_actor = self.gym.find_actor_rigid_body_index(env_ptr, indenter_actor_handle, indenter_rb_name, gymapi.DOMAIN_ACTOR)
                indenter_rb_shape_indices = self.gym.get_actor_rigid_body_shape_indices(env_ptr, indenter_actor_handle)
                indenter_rb_shape_id_actor = indenter_rb_shape_indices[indenter_rb_id_actor].start
                indenter_rb_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, indenter_actor_handle)
                # TO-DO: Fix this potential bug: if there is multiple indentors; the global index will be overwritten by the last one
                # self.sdf_shape_global_ids_per_env[env_id, 0] = indenter_rb_shape_props[indenter_rb_shape_id_actor].global_index
                sdf_shape_global_ids_per_env[env_id, 0] = indenter_rb_shape_props[indenter_rb_shape_id_actor].global_index
            sdf = None
        else:
            raise NotImplementedError

        return sdf, indenter_mesh, indenter_mesh_local_tf, sdf_shape_global_ids_per_env

    def initialize_penalty_based_tactile(self, num_divs):
        # Use one of the sensor configs to set-up the force-field computation
        # Assumption here is that there is a single type of sensor, though there can be multiple instances e.g. left, right sensor
        # Store per-force-field data in dicts
        self.tactile_pos_local_dict = {}
        self.tactile_quat_local_dict = {}
        self.sdf_dict = {}
        self.indenter_mesh_dict = {}
        self.indenter_mesh_local_tf_dict = {}
        self.sdf_shape_global_ids_per_env_dict = {}
        
        for key, sensor_config in self.tactile_shear_field_configs_dict.items():
            # sensor_config = list(self.tactile_shear_field_configs_dict.values())[0]
            # if key == "tactile_force_field_right_ppball":
            #     print(f"Skipping {key} as it is not implemented yet.")
            #     continue
            indenter_link_rb_id = sensor_config['indenter_link_rb_id']
            # Generate tactile points
            tactile_pos_local, tactile_quat_local = self.generate_tactile_points(
                elastomer_parent_urdf_path=sensor_config['elastomer_parent_urdf_path'],
                elastomer_link_name=sensor_config['elastomer_link_name'],
                elastomer_tip_link_name=sensor_config['elastomer_tip_link_name'],
                elastomer_actor_name=sensor_config['elastomer_actor_name'],
                num_divs=num_divs, visualize=False)
            self.tactile_pos_local_dict[indenter_link_rb_id] = tactile_pos_local
            self.tactile_quat_local_dict[indenter_link_rb_id] = tactile_quat_local
            
            print(f"🏓Indenter link {indenter_link_rb_id} has {tactile_pos_local.shape[0]} tactile points.")

            # Initialize SDF and mesh
            sdf, indenter_mesh, indenter_mesh_local_tf, sdf_shape_global_ids_per_env = self.load_sdf_oracle_of_indenter(
                indenter_urdf_path=sensor_config['indenter_urdf_path'],
                indenter_actor_name=sensor_config['indenter_actor_name'],
                indenter_rb_name=sensor_config['indenter_link_name'])
            self.sdf_dict[indenter_link_rb_id] = sdf
            self.indenter_mesh_dict[indenter_link_rb_id] = indenter_mesh
            self.indenter_mesh_local_tf_dict[indenter_link_rb_id] = indenter_mesh_local_tf
            self.sdf_shape_global_ids_per_env_dict[indenter_link_rb_id] = sdf_shape_global_ids_per_env

        # Optionally, set a default (for backward compatibility)
        if self.tactile_pos_local_dict:
            first_key = list(self.tactile_pos_local_dict.keys())[0]
            self.tactile_pos_local = self.tactile_pos_local_dict[first_key]
            self.tactile_quat_local = self.tactile_quat_local_dict[first_key]
            # self.sdf = self.sdf_dict[first_key]
            # self.indenter_mesh = self.indenter_mesh_dict[first_key]
            # self.indenter_mesh_local_tf = self.indenter_mesh_local_tf_dict[first_key]

        # initialize coefficients
        self.tactile_kn = 1.
        self.tactile_damping = 0.003
        self.tactile_mu = 2.
        self.tactile_kt = 0.1 # 0.5

        # --- For vt_world bias lookup ---
        self.NUM_BINS = 50
        self.Y_ANGVEL_RANGE = (-0.25, 0.25)
        self.lookup_bins = np.linspace(self.Y_ANGVEL_RANGE[0], self.Y_ANGVEL_RANGE[1], self.NUM_BINS + 1)
        self.y_angvel_to_vtworld = defaultdict(list)

        self.vt_lookup_save_counter = 0
        self.vt_lookup_save_interval = 500  # Save every 200 frames

        self.load_vt_lookup_table(LOOKUP_TABLE_PATH)

    def query_collision(self, sdf, tf_sdf, sdf_linvel_world, sdf_angvel_world, points_world, velocity_world):
        """
        Query collisions in the SDF.

        Args:
            sdf: Signed-distance field of the object.
            tf_sdf: (pos, quat) of the object/SDF frame.
            sdf_linvel_world: Linear velocity of the SDF object in the world frame.
            sdf_angvel_world: Angular velocity of the SDF object in the world frame.
            points_world: Points in the world frame.
            velocity_world: Velocities of the points in the world frame.

        Returns:
            tuple: Depth, depth_dot, normal, and vt (all in the world frame).
        """
        num_points_per_env = points_world.shape[1]

        tf_sdf = (tf_sdf[0].unsqueeze(1).expand([self.num_envs, num_points_per_env, 4]),
                  tf_sdf[1].unsqueeze(1).expand([self.num_envs, num_points_per_env, 3]))
        sdf_linvel_world = sdf_linvel_world.unsqueeze(1).expand([self.num_envs, num_points_per_env, 3])
        sdf_angvel_world = sdf_angvel_world.unsqueeze(1).expand([self.num_envs, num_points_per_env, 3])

        tf_sdf_inv = tu.tf_inverse(tf_sdf[0], tf_sdf[1])

        # compute points in the object frame
        points_sdf = tu.tf_apply(tf_sdf_inv[0], tf_sdf_inv[1], points_world)
        # compute depth
        if self.sdf_tool == 'trimesh':
            collision_mask_flatten, depth_flatten, normal_flatten_sdf = \
                self.query_distance_and_normal_with_trimesh(sdf, points_sdf.view(-1, 3))
        elif self.sdf_tool == 'pysdf':
            collision_mask_flatten, depth_flatten, normal_flatten_sdf = \
                self.query_distance_and_normal_with_pysdf(sdf, points_sdf.view(-1, 3))
        elif self.sdf_tool == 'physx':
            collision_mask_flatten, depth_flatten, normal_flatten_sdf = \
                self.query_distance_and_normal_with_physx(points_sdf.view(self.num_envs, 1, -1, 3))
        else:
            raise NotImplementedError

        depth = depth_flatten.reshape(points_world.shape[:-1])
        depth = depth.clamp(min=0., max=None)

        # compute other returned values
        normal_world = torch.zeros(points_world.shape, device=self.device)
        depth_dot = torch.zeros(points_world.shape[:-1], device=self.device)
        vt_world = torch.zeros(points_world.shape, device = self.device)
        vt_world_corrected = torch.zeros(points_world.shape, device = self.device)

        if collision_mask_flatten.sum() > 0:
            normal_sdf = normal_flatten_sdf.reshape(normal_world.shape)
            normal_world = tu.quat_apply(tf_sdf[0], normal_sdf)

            '''
            x = R.T (xw - p)
            xdot = Rdot.T (xw - p) + R.T (xwdot - pdot)
                 = R.T [w].T (xw - p) + R.T (xwdot - pdot)
                 = R.T (-[w] (xw - p) + xwdot - pdot)
                 = R.T ((xw - p) x [w] + xwdot - pdot)
            '''
            velocity_sdf = tu.quat_apply(tf_sdf_inv[0],
                                         torch.cross(points_world - tf_sdf[1], sdf_angvel_world, dim = -1) +
                                         velocity_world - sdf_linvel_world)

            '''
            ddot = dd/dx * dx/dt = n.T * xdot
            '''
            depth_dot = torch.sum(normal_sdf * velocity_sdf, dim = -1)

            '''
            xc_world = R * xc + p
            xcdot_world = [w] R xc + R xcdot + pdot
                        = [w] R xc + pdot
            '''
            closest_points_sdf = points_sdf + depth.unsqueeze(-1) * normal_sdf
            closest_points_velocity_world = torch.cross(sdf_angvel_world, tu.quat_apply(tf_sdf[0], closest_points_sdf)) + sdf_linvel_world
            relative_velocity_world = velocity_world - closest_points_velocity_world

            vt_world = relative_velocity_world - normal_world * torch.sum(normal_world * relative_velocity_world, dim = -1, keepdim = True)
            
            
            # Compute the norm for each tactile point (shape: [1, 140])
            elastomer_link_id = 16
            elastomer_angvel_world = self.body_angvel[:, elastomer_link_id]
            # Example usage inside a method of TactileFieldSensor
            y_angvel = elastomer_angvel_world[0, 1].cpu().item()
            
            # vt_world_np = vt_world[0].cpu().numpy()  # shape (num_points, 3)
            # self.update_vt_lookup_table(y_angvel, vt_world_np)
            # self.vt_lookup_save_counter += 1
            # print(f"vt_lookup_save_counter: {self.vt_lookup_save_counter}, y_angvel: {y_angvel:.6f}, vt_world norm: {np.linalg.norm(vt_world_np, axis=-1).mean():.6f}")
            # if self.vt_lookup_save_counter % self.vt_lookup_save_interval == 0:
            #     self.save_vt_lookup_table("my_vt_lookup_table_impedance.pkl")

            # To get the bias for a given y_angvel:
            vt_bias_np = self.get_vt_bias(y_angvel)   # shape (num_points, 3)
            # Convert vt_bias to torch tensor on the same device and dtype as vt_world
            vt_bias = torch.tensor(vt_bias_np, device=vt_world.device, dtype=vt_world.dtype)
            # print(f"vt_bias_np: {np.linalg.norm(vt_bias_np, axis=-1).mean():.6f}")
            vt_world_corrected = vt_world - vt_bias.unsqueeze(0)

        return depth, depth_dot, normal_world, vt_world_corrected

    # def query_collision(self, sdf, tf_sdf, sdf_linvel_world, sdf_angvel_world, points_world, velocity_world, elastomer_link_id):
    #     """
    #     Query collisions in the SDF.

    #     Args:
    #         sdf: Signed-distance field of the object.
    #         tf_sdf: (pos, quat) of the object/SDF frame.
    #         sdf_linvel_world: Linear velocity of the SDF object in the world frame.
    #         sdf_angvel_world: Angular velocity of the SDF object in the world frame.
    #         points_world: Points in the world frame.
    #         velocity_world: Velocities of the points in the world frame.

    #     Returns:
    #         tuple: Depth, depth_dot, normal, and vt (all in the world frame).
    #     """
    #     num_points_per_env = points_world.shape[1]

    #     tf_sdf = (tf_sdf[0].unsqueeze(1).expand([self.num_envs, num_points_per_env, 4]),
    #               tf_sdf[1].unsqueeze(1).expand([self.num_envs, num_points_per_env, 3]))
    #     sdf_linvel_world = sdf_linvel_world.unsqueeze(1).expand([self.num_envs, num_points_per_env, 3])
    #     sdf_angvel_world = sdf_angvel_world.unsqueeze(1).expand([self.num_envs, num_points_per_env, 3])

    #     tf_sdf_inv = tu.tf_inverse(tf_sdf[0], tf_sdf[1])

    #     # compute points in the object frame
    #     points_sdf = tu.tf_apply(tf_sdf_inv[0], tf_sdf_inv[1], points_world)

    #     # compute depth
    #     if self.sdf_tool == 'trimesh':
    #         print('trimesh')
    #         collision_mask_flatten, depth_flatten, normal_flatten_sdf = \
    #             self.query_distance_and_normal_with_trimesh(sdf, points_sdf.view(-1, 3))
    #     elif self.sdf_tool == 'pysdf':
    #         print('pysdf')
    #         collision_mask_flatten, depth_flatten, normal_flatten_sdf = \
    #             self.query_distance_and_normal_with_pysdf(sdf, points_sdf.view(-1, 3))
    #     elif self.sdf_tool == 'physx':
    #         print('physx')
    #         collision_mask_flatten, depth_flatten, normal_flatten_sdf = \
    #             self.query_distance_and_normal_with_physx(points_sdf.view(self.num_envs, 1, -1, 3))
    #     else:
    #         raise NotImplementedError

    #     depth = depth_flatten.reshape(points_world.shape[:-1])
    #     depth = depth.clamp(min=0., max=None)

    #     # compute other returned values
    #     normal_world = torch.zeros(points_world.shape, device=self.device)
    #     depth_dot = torch.zeros(points_world.shape[:-1], device=self.device)
    #     vt_world = torch.zeros(points_world.shape, device = self.device)
    #     tactile_displace_world = torch.zeros(points_world.shape, device = self.device)

    #     if collision_mask_flatten.sum() > 0:
    #         # print(f"collision_mask_flatten: {collision_mask_flatten}")
    #         normal_sdf = normal_flatten_sdf.reshape(normal_world.shape)
    #         normal_world = tu.quat_apply(tf_sdf[0], normal_sdf)

    #         '''
    #         x = R.T (xw - p)
    #         xdot = Rdot.T (xw - p) + R.T (xwdot - pdot)
    #              = R.T [w].T (xw - p) + R.T (xwdot - pdot)
    #              = R.T (-[w] (xw - p) + xwdot - pdot)
    #              = R.T ((xw - p) x [w] + xwdot - pdot)
    #         '''
    #         velocity_sdf = tu.quat_apply(tf_sdf_inv[0],
    #                                      torch.cross(points_world - tf_sdf[1], sdf_angvel_world, dim = -1) +
    #                                      velocity_world - sdf_linvel_world)

    #         '''
    #         ddot = dd/dx * dx/dt = n.T * xdot
    #         '''
    #         depth_dot = torch.sum(normal_sdf * velocity_sdf, dim = -1)

    #         '''
    #         xc_world = R * xc + p
    #         xcdot_world = [w] R xc + R xcdot + pdot
    #                     = [w] R xc + pdot
    #         '''
    #         closest_points_sdf = points_sdf + depth.unsqueeze(-1) * normal_sdf
    #         closest_points_velocity_world = torch.cross(sdf_angvel_world, tu.quat_apply(tf_sdf[0], closest_points_sdf)) + sdf_linvel_world
    #         relative_velocity_world = velocity_world - closest_points_velocity_world

    #         vt_world = relative_velocity_world - normal_world * torch.sum(normal_world * relative_velocity_world, dim = -1, keepdim = True)
    #         # print(f"closest_points_sdf, {closest_points_sdf.shape}")


    #         # new feature developed by Quan
    #         # print(f"{tf_sdf[0].shape}, {tf_sdf[1].shape}")
    #         elastomer_pos = self.body_pos[:, elastomer_link_id] # shape (1, 3)
    #         elastomer_quat = self.body_quat[:, elastomer_link_id] # shape (1, 4)
    #         # Compute the inverse of the elastomer pose
    #         elastomer_quat_inv, elastomer_pos_inv = tf_inverse(elastomer_quat, elastomer_pos)
    #         # Compute the object pose in the elastomer frame
    #         quat_rel, pos_rel = tf_combine(elastomer_quat_inv, elastomer_pos_inv, tf_sdf[0][:, 0, :], tf_sdf[1][:, 0, :])
    #         # print(f"quat_rel: {quat_rel}, {quat_rel.shape}")
    #         # print(f"pos_rel: {pos_rel}, {pos_rel.shape}")
            
    #         # Expand quat_rel and pos_rel to match (1, 140, *)
    #         quat_rel_exp = quat_rel.unsqueeze(1).expand(self.num_envs, num_points_per_env, 4)  # (1, 140, 4)
    #         pos_rel_exp = pos_rel.unsqueeze(1).expand(self.num_envs, num_points_per_env, 3)    # (1, 140, 3)
    #         # Transform points into the elastomer frame
    #         closest_points_elastomer = tu.quat_apply(quat_rel_exp, closest_points_sdf) + pos_rel_exp  # (1, 140, 3)
    #         tactile_points_pos_tmp = self.tactile_pos_local.unsqueeze(0).expand(self.num_envs, num_points_per_env, 3)
    #         # print(f"closest_points, and tactile_pos_local: {closest_points_elastomer.shape}, {tactile_points_pos_tmp.shape}")
    #         tactile_points_displace = closest_points_elastomer - tactile_points_pos_tmp
    #         # Zero out Y component (index 1)
    #         tactile_points_displace[:, :, 1] = 0.0
            
    #         # Expand quaternion to match (1, N, 4) if needed
    #         quat_exp = elastomer_quat.unsqueeze(1).expand(self.num_envs, num_points_per_env, 4)
    #         # Convert displacement vector to world frame by applying only rotation
    #         tactile_displace_world = tu.quat_apply(quat_exp, tactile_points_displace)  # shape (1, N, 3)


    #         # TO-DO: enable relative rotation if needed
    #         # # print(f"normal_sdf: {normal_sdf}, {normal_sdf.shape}")
    #         # # Suppose quat_rel is of shape (1, 4)
    #         # roll, pitch, yaw = get_euler_xyz(quat_rel)
    #         # pitch = ((pitch + np.pi) % (2 * np.pi)) - np.pi
    #         # # Convert pitch (Y-axis rotation) to degrees
    #         # pitch_deg = pitch * 180.0 / np.pi

    #     return depth, depth_dot, normal_world, vt_world, tactile_displace_world

    def post_process_shear_field_configs(self, tactile_shear_field_configs):
        def get_link_handle(actor_name, link_name):
            link_handle = self.gym.find_actor_rigid_body_handle(
                self.env_ptrs[0], self.actor_handles[actor_name], link_name)
            return link_handle

        for tactile_shear_field_config in tactile_shear_field_configs:
            tactile_shear_field_config['elastomer_link_rb_id'] = get_link_handle(
                tactile_shear_field_config['elastomer_actor_name'], tactile_shear_field_config['elastomer_link_name'])
            tactile_shear_field_config['indenter_link_rb_id'] = get_link_handle(
                tactile_shear_field_config['indenter_actor_name'], tactile_shear_field_config['indenter_link_name'])

        tactile_shear_field_configs_dict = dict()
        for config in tactile_shear_field_configs:
            # if config['name'] == "tactile_force_field_right_ppball":
            #     print(f"Skipping {config['name']} as it is not implemented yet.")
            #     continue
            tactile_shear_field_configs_dict[config['name']] =  config
        return tactile_shear_field_configs_dict

    def get_tactile_shear_force_fields(self):
        tactile_force_field = dict()
        for key, config in self.tactile_shear_field_configs_dict.items():
            # if key == "tactile_force_field_right_ppball":
            #     print(f"Skipping {key} as it is not implemented yet.")
            #     continue
            indenter_link_id = config['indenter_link_rb_id']
            elastomer_link_id = config['elastomer_link_rb_id']
            # print(f"indenter_link_id: {indenter_link_id}, elastomer_link_id: {elastomer_link_id}")
            result = self.get_penalty_based_tactile_forces(indenter_link_id, elastomer_link_id)
            tactile_force_field[key] = result
        return tactile_force_field

    def get_tactile_points_velocities(self, elastomer_link_id):
        elastomer_angvel_world = self.body_angvel[:, elastomer_link_id]
        elastomer_linvel_world = self.body_linvel[:, elastomer_link_id]
        elastomer_quat_world = self.body_quat[:, elastomer_link_id]
        num_tactile_points = self.tactile_pos_local.shape[0]
        tactile_velocity_world = torch.cross(
            elastomer_angvel_world.unsqueeze(1).expand((self.num_envs, num_tactile_points, 3)),
            tu.quat_apply(elastomer_quat_world.unsqueeze(1).expand((self.num_envs, num_tactile_points, 4)),
                       self.tactile_pos_local.expand((self.num_envs, num_tactile_points, 3)))) \
                                 + elastomer_linvel_world.unsqueeze(1).expand((self.num_envs, num_tactile_points, 3))
        return tactile_velocity_world

    def get_penalty_based_tactile_forces(self, indenter_link_id, elastomer_link_id):
        """
        Get the penalty-based tactile forces.

        Ref: https://openreview.net/forum?id=6BIffCl6gsM

        Args:
            indenter_link_id (int): ID of the indenter link.
            elastomer_link_id (int): ID of the elastomer link.

        Returns:
            tuple: Interpenetration depth, tactile normal force, and tactile shear force.
        """

        # print(f"indenter_link_id: {indenter_link_id}, elastomer_link_id: {elastomer_link_id}")
        self.tactile_pos_local = self.tactile_pos_local_dict[indenter_link_id]
        self.tactile_quat_local = self.tactile_quat_local_dict[indenter_link_id]
        self.sdf = self.sdf_dict[indenter_link_id]
        self.indenter_mesh = self.indenter_mesh_dict[indenter_link_id]
        self.indenter_mesh_local_tf = self.indenter_mesh_local_tf_dict[indenter_link_id]
        self.sdf_shape_global_ids_per_env = self.sdf_shape_global_ids_per_env_dict[indenter_link_id]

        # acquire sdf related variables
        sdf_tf = tf_combine(self.body_quat[:, indenter_link_id],
                            self.body_pos[:, indenter_link_id],
                            self.indenter_mesh_local_tf[0].expand(self.num_envs, 4),
                            self.indenter_mesh_local_tf[1].expand(self.num_envs, 3))

        sdf_angvel_world, sdf_linvel_world = self.body_angvel[:, indenter_link_id], self.body_linvel[:, indenter_link_id]
        # print(f"sdf_linvel_world: {sdf_linvel_world}, {indenter_link_id}")

        self.tactile_pos_world, self.tactile_quat_world = self.get_tactile_points_in_world(
            self.tactile_pos_local, self.tactile_quat_local, elastomer_link_id
        )
        # tactile_velocity_world = torch.zeros_like(self.tactile_pos_world) # NOTE [Jie]: now assume fingers are fixed
        tactile_velocity_world = self.get_tactile_points_velocities(elastomer_link_id)
        # print(tactile_velocity_world.abs().sum())
        
        depth, depth_dot, normal_world, vt_world = self.query_collision(self.sdf, sdf_tf, sdf_linvel_world, sdf_angvel_world,
                                                                        self.tactile_pos_world, tactile_velocity_world)

        # tactile_points_displace = tactile_points_displace
        # print(f"tactile_points_displace: {tactile_points_displace}, {tactile_points_displace.shape}")
        # print(f"Rotation around Y-axis: {pitch_deg.item():.2f} degrees")

        # DEPTH_THRESHOLD = 0.004  # meters
        # VT_SAVE_INTERVAL = 1000  # frames
        # # Compute total contact depth norm
        # contact_depth = depth.norm(dim=-1).item()
        # print(f"contact_depth: {contact_depth}")
        
        # if contact_depth > DEPTH_THRESHOLD:
        #     print("record")
        #     # --- Record vt_world ---
        #     global vt_world_history
        #     if 'vt_world_history' not in globals():
        #         vt_world_history = []

        #     # Clone and detach for safe storage
        #     vt_world_history.append(vt_world.clone().detach().cpu())

        #     print(f"len history: {len(vt_world_history)}")

        #     # Save to disk if enough frames collected
        #     if len(vt_world_history) >= VT_SAVE_INTERVAL:
                
        #         save_vt_world_history_to_disk()
        #         vt_world_history.clear()
        
        # print(f"depth: {depth.norm(dim=-1)}")

        # # --- Apply low-pass filter ---
        # if not hasattr(self, 'prev_vt_world'):
        #     self.prev_vt_world = vt_world.clone()

        # alpha = getattr(self, 'vt_filter_alpha', 0.386)  # default alpha = 0.1 if not defined
        # vt_world = alpha * vt_world + (1 - alpha) * self.prev_vt_world
        # self.prev_vt_world = vt_world.clone()
       
        # # --- High-pass filter vt_world to remove <25Hz ---
        # if not hasattr(self, 'prev_vt_world_input'):
        #     self.prev_vt_world_input = vt_world.clone()
        #     self.prev_vt_world_filtered = torch.zeros_like(vt_world)

        # # Calculate alpha for high-pass filter
        # fc = 25.0      # cutoff frequency in Hz
        # fs = 100.0     # sampling rate in Hz
        # rc = 1.0 / (2 * np.pi * fc)
        # dt = 1.0 / fs
        # alpha = rc / (rc + dt)  # derived from the RC circuit analogy

        # # High-pass filter equation
        # vt_world_filtered = alpha * (self.prev_vt_world_filtered + vt_world - self.prev_vt_world_input)

        # # Update buffers for next step
        # self.prev_vt_world_input = vt_world.clone()
        # self.prev_vt_world_filtered = vt_world_filtered.clone()

        # # Replace vt_world with high-passed version
        # vt_world = vt_world_filtered

        # --- Apply exponential decay filter to vt_world ---
        # vt_norms = vt_world.norm(dim=-1)  # shape (1, N)
        # decay_factor = 1.0 - torch.exp(-20.0 * (vt_norms - 0.03).clamp(min=0.0))  # smooth decay
        # decay_factor = decay_factor.unsqueeze(-1)  # shape (1, N, 1)

        # vt_world = vt_world * decay_factor

       
        # # --- Apply thresholding ---
        # vt_norms = vt_world.norm(dim=-1)
        # # Create a mask for small tangential velocities (True where norm < 0.04)
        # mask = vt_norms < 0.02  # 0.004 for high pass filter approach, 0.02 for pure thresholding approach
        # # Zero out vt_world at those locations
        # vt_world[mask] = 0.0

        # # --- Apply integration for displacement ---
        # if not hasattr(self, 'displacment_world'):
        #     self.displacment_world = torch.zeros_like(vt_world)
        # self.displacment_world = self.displacment_world + vt_world*0.1


        # # --- NEW: Plot mean vt_world xyz components in real-time ---
        # global vt_world_buffer, plot_counter
        # # Compute mean of each xyz component over all contact points (shape: (3,))
        # vt_mean_xyz = vt_world.mean(dim=(0, 1)).cpu().numpy()  # mean over (env, point), shape (3,)
        # vt_world_buffer.append(vt_mean_xyz)
        # if len(vt_world_buffer) > BUFFER_SIZE:
        #     vt_world_buffer.pop(0)

        # plot_counter += 1
        # if plot_counter % PLOT_INTERVAL == 0:
        #     plt.ion()
        #     plt.clf()
        #     plt.title("Real-Time vt_world Mean XYZ Components")
        #     plt.xlabel("Timestep")
        #     plt.ylabel("Mean vt_world (m/s)")
        #     vt_arr = np.array(vt_world_buffer)  # shape: (buffer_size, 3)
        #     if vt_arr.shape[0] > 0:
        #         plt.plot(vt_arr[:, 0], label='vx')
        #         plt.plot(vt_arr[:, 1], label='vy')
        #         plt.plot(vt_arr[:, 2], label='vz')
        #         plt.legend()
        #         plt.pause(0.001)

        # [UNCHANGED CODE CONTINUES]

        # compute tactile forces in world frame
        fc_norm = self.tactile_kn * depth
        fc_world = fc_norm.unsqueeze(-1) * normal_world

        # displacemnt_norm = tactile_points_displace.norm(dim=-1)
        # ft_static_norm = self.tactile_kt * displacemnt_norm
        # ft_dynamic_norm = self.tactile_mu * fc_norm
        # ft_world = -torch.minimum(ft_static_norm, ft_dynamic_norm).unsqueeze(-1) * tactile_points_displace / displacemnt_norm.clamp(min=1e-9).unsqueeze(-1)

        # displacemnt_norm = self.displacment_world.norm(dim=-1)
        # ft_static_norm = self.tactile_kt * displacemnt_norm
        # ft_dynamic_norm = self.tactile_mu * fc_norm
        # ft_world = -torch.minimum(ft_static_norm, ft_dynamic_norm).unsqueeze(-1) * self.displacment_world / displacemnt_norm.clamp(min=1e-9).unsqueeze(-1)

        vt_norm = vt_world.norm(dim=-1)
        ft_static_norm = self.tactile_kt * vt_norm
        ft_dynamic_norm = self.tactile_mu * fc_norm
        ft_world = -torch.minimum(ft_static_norm, ft_dynamic_norm).unsqueeze(-1) * vt_world / vt_norm.clamp(min=1e-9).unsqueeze(-1)

        # print(f"ft_dynamic norm: {ft_dynamic_norm}")

        tactile_force_world = fc_world + ft_world
        quat_tactile_inv = tu.quat_conjugate(self.tactile_quat_world)
        tactile_force_tactile = tu.quat_apply(quat_tactile_inv, tactile_force_world)

        tactile_normal_axis = torch.tensor([0., 1., 0.], device=self.device)
        tactile_shear_x_axis = torch.tensor([-1., 0., 0.], device=self.device)
        tactile_shear_y_axis = torch.tensor([0., 0., 1.], device=self.device)

        tactile_normal_force = -(tactile_normal_axis.view(1, 1, -1) * tactile_force_tactile).sum(-1)
        tactile_shear_force_x = (tactile_shear_x_axis.view(1, 1, -1) * tactile_force_tactile).sum(-1)
        tactile_shear_force_y = (tactile_shear_y_axis.view(1, 1, -1) * tactile_force_tactile).sum(-1)
        tactile_shear_force = torch.cat((tactile_shear_force_x.unsqueeze(-1), tactile_shear_force_y.unsqueeze(-1)), dim=-1)

        return depth, tactile_normal_force, tactile_shear_force



    # def get_penalty_based_tactile_forces(self, indenter_link_id, elastomer_link_id):
    #     """
    #     Get the penalty-based tactile forces.

    #     Ref: https://openreview.net/forum?id=6BIffCl6gsM

    #     Args:
    #         indenter_link_id (int): ID of the indenter link.
    #         elastomer_link_id (int): ID of the elastomer link.

    #     Returns:
    #         tuple: Interpenetration depth, tactile normal force, and tactile shear force.
    #     """

    #     # acquire sdf related variables
    #     sdf_tf = tf_combine(self.body_quat[:, indenter_link_id],
    #                         self.body_pos[:, indenter_link_id],
    #                         self.indenter_mesh_local_tf[0].expand(self.num_envs, 4),
    #                         self.indenter_mesh_local_tf[1].expand(self.num_envs, 3))

    #     sdf_angvel_world, sdf_linvel_world = self.body_angvel[:, indenter_link_id], self.body_linvel[:, indenter_link_id]

    #     self.tactile_pos_world, self.tactile_quat_world = self.get_tactile_points_in_world(
    #         self.tactile_pos_local, self.tactile_quat_local, elastomer_link_id
    #     )
    #     # tactile_velocity_world = torch.zeros_like(self.tactile_pos_world) # NOTE [Jie]: now assume fingers are fixed
    #     tactile_velocity_world = self.get_tactile_points_velocities(elastomer_link_id)
    #     # print(tactile_velocity_world.abs().sum())

    #     depth, depth_dot, normal_world, vt_world = self.query_collision(self.sdf, sdf_tf, sdf_linvel_world, sdf_angvel_world,
    #                                                self.tactile_pos_world, tactile_velocity_world)

    #     # compute tactile forces in world frame
    #     '''compute contact force'''
    #     fc_norm = self.tactile_kn * depth #- self.tactile_damping * depth_dot * depth
    #     fc_world = fc_norm.unsqueeze(-1) * normal_world

    #     '''compute frictional force'''
    #     vt_norm = vt_world.norm(dim=-1)
    #     ft_static_norm = self.tactile_kt * vt_norm
    #     ft_dynamic_norm = self.tactile_mu * fc_norm
    #     ft_world = -torch.minimum(ft_static_norm, ft_dynamic_norm).unsqueeze(-1) * vt_world / vt_norm.clamp(min=1e-9, max=None).unsqueeze(-1)

    #     # print("stacti norm, dynamic norm, ft_world", ft_static_norm, ft_dynamic_norm, ft_world)

    #     '''net tactile force'''
    #     tactile_force_world = fc_world + ft_world

    #     '''tactile force in tactile frame'''
    #     quat_tactile_inv = tu.quat_conjugate(self.tactile_quat_world)
    #     tactile_force_tactile = tu.quat_apply(quat_tactile_inv, tactile_force_world)

    #     # tactile_normal_force = -tactile_force_tactile[..., 2]
    #     # tactile_shear_force = tactile_force_tactile[..., 0:2]
    #     tactile_normal_axis = torch.tensor([0., 1., 0.], device=self.device)
    #     tactile_shear_x_axis = torch.tensor([-1., 0., 0.], device=self.device)
    #     tactile_shear_y_axis = torch.tensor([0., 0., 1.], device=self.device)
    #     # tactile_normal_force = -tactile_force_tactile[..., 1] # NOTE: the tactile frame has y as normal direction, to be changed
    #     # tactile_shear_force = tactile_force_tactile[..., 0:3:2]
    #     tactile_normal_force = -(tactile_normal_axis.view(1, 1, -1) * tactile_force_tactile).sum(-1)
    #     tactile_shear_force_x = (tactile_shear_x_axis.view(1, 1, -1) * tactile_force_tactile).sum(-1)
    #     tactile_shear_force_y = (tactile_shear_y_axis.view(1, 1, -1) * tactile_force_tactile).sum(-1)
    #     tactile_shear_force = torch.cat((tactile_shear_force_x.unsqueeze(-1), tactile_shear_force_y.unsqueeze(-1)), dim=-1)

    #     # print(f"🚀tactile_normal_force, min:{torch.min(tactile_normal_force)}, max:{torch.max(tactile_normal_force)}")
    #     # print(f"🚀tactile_shear_force_x, min:{torch.min(tactile_shear_force_x)}, max:{torch.max(tactile_shear_force_x)}")
    #     # print(f"🚀tactile_shear_force_y, min:{torch.min(tactile_shear_force_y)}, max:{torch.max(tactile_shear_force_y)}")

    #     return depth, tactile_normal_force, tactile_shear_force


    def save_vt_lookup_table(self, path):
        # Convert defaultdict to dict for saving
        data = {
            'NUM_BINS': self.NUM_BINS,
            'Y_ANGVEL_RANGE': self.Y_ANGVEL_RANGE,
            'lookup_bins': self.lookup_bins,
            'y_angvel_to_vtworld': dict(self.y_angvel_to_vtworld)
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Lookup table saved to {path}")

    def load_vt_lookup_table(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.NUM_BINS = data['NUM_BINS']
        self.Y_ANGVEL_RANGE = data['Y_ANGVEL_RANGE']
        self.lookup_bins = data['lookup_bins']
        # Restore as defaultdict(list)
        self.y_angvel_to_vtworld = defaultdict(list, data['y_angvel_to_vtworld'])
        print(f"Lookup table loaded from {path}")

    def update_vt_lookup_table(self, y_angvel, vt_world):
            """
            y_angvel: float (the y angular velocity)
            vt_world: np.ndarray of shape (num_points, 3)
            """
            bin_idx = np.digitize([y_angvel], self.lookup_bins)[0] - 1
            if 0 <= bin_idx < self.NUM_BINS:
                self.y_angvel_to_vtworld[bin_idx].append(vt_world.copy())

    def get_vt_bias(self, y_angvel):
        """
        y_angvel: float
        Returns: np.ndarray of shape (num_points, 3) or zeros if no data
        """
        bin_idx = np.digitize([y_angvel], self.lookup_bins)[0] - 1
        if 0 <= bin_idx < self.NUM_BINS and len(self.y_angvel_to_vtworld[bin_idx]) > 0:
            vt_stack = np.stack(self.y_angvel_to_vtworld[bin_idx], axis=0)  # shape: (N, num_points, 3)
            return vt_stack.mean(axis=0)
        else:
            # Return zeros if no data for this bin
            return np.zeros((self.tactile_pos_local.shape[0], 3))

def suppress_angular_velocity_noise(angvel_world, threshold_deg=2.0):
    """
    Suppress small angular velocity if likely due to noise.

    Args:
        angvel_world (torch.Tensor): Angular velocity tensor of shape (1, 3) or (B, 3), in radians/sec.
        threshold_deg (float): Threshold in degrees/sec below which motion is considered noise.

    Returns:
        torch.Tensor: Filtered angular velocity (same shape), with small values zeroed.
    """
    # Convert threshold to radians
    threshold_rad = threshold_deg * np.pi / 180.0

    # Compute norm
    angvel_norm = angvel_world.norm(dim=-1, keepdim=True)  # shape (1, 1) or (B, 1)

    # Create mask: True where angvel is significant
    keep_mask = angvel_norm >= threshold_rad

    # Zero out angular velocity where it's likely just noise
    angvel_filtered = angvel_world * keep_mask

    return angvel_filtered

from datetime import datetime
vt_world_history = []

def save_vt_world_history_to_disk():
    if len(vt_world_history) == 0:
        print("No vt_world data to save.")
        return
    vt_tensor = torch.stack(vt_world_history, dim=0)  # shape: [T, 1, N, 3]
    time_str = datetime.now().strftime("%m%d_%H%M%S")
    path = f"vt_world_{time_str}.npy"
    np.save(path, vt_tensor.numpy())
    print(f"[SAVED] vt_world history → {path}")