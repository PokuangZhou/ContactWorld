from gym import spaces

import isaacgym
# import isaacgymenvs
# from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.tasks.tacsl.tacsl_task_insertion import TacSLTaskInsertion
from isaacgymenvs.tasks.tacsl.tacsl_task_usb import TacSLTaskUSB
# from isaacgymenvs.tasks.tacsl.tacsl_task_power import TacSLTaskPower
from isaacgymenvs.tasks.tacsl.tacsl_task_bolt_nut import TacSLTaskBoltNut
# from isaacgymenvs.tasks.tacsl.tacsl_task_bulb import TacSLTaskBulb
# from isaacgymenvs.tasks.tacsl.tacsl_task_sorting_normal import TacSLTaskSortingNormal
# from isaacgymenvs.tasks.tacsl.tacsl_task_barbed_spike import TacSLTaskBarbedSpike
# from isaacgymenvs.tasks.tacsl.tacsl_task_barbed_flat import TacSLTaskBarbedFlat
# from isaacgymenvs.tasks.tacsl.tacsl_task_lidded_loose_plug import TacSLTaskLiddedLoose
# from isaacgymenvs.tasks.tacsl.tacsl_task_search import TacSLTaskSearch
# from isaacgymenvs.tasks.tacsl.tacsl_task_valve import TacSLTaskValve
# from isaacgymenvs.tasks.tacsl.tacsl_task_valve_big import TacSLTaskValveBig
# from isaacgymenvs.tasks.tacsl.tacsl_task_plug import TacSLTaskPlug
from isaacgymenvs.tasks.tacsl.tacsl_task_bulb import TacSLTaskBulb

from isaacgymenvs.utils.reformat import omegaconf_to_dict

from utils.shear_tactile_viz_utils import visualize_tactile_shear_image, visualize_penetration_depth
import torch
import numpy as np
import cv2
import random
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# Mappings from strings to environments
isaacgym_task_map = {
    "TacSLTaskInsertion": TacSLTaskInsertion,
    "TacSLTaskUSB": TacSLTaskUSB,
    # "TacSLTaskPower": TacSLTaskPower,
    "TacSLTaskBoltNut": TacSLTaskBoltNut,
    # "TacSLTaskBulb": TacSLTaskBulb,
    # "TacSLTaskSortingNormal": TacSLTaskSortingNormal,
    # "TacSLTaskBarbedSpike": TacSLTaskBarbedSpike,
    # "TacSLTaskBarbedFlat": TacSLTaskBarbedFlat,
    # "TacSLTaskLiddedLoose": TacSLTaskLiddedLoose,
    # "TacSLTaskSearch": TacSLTaskSearch,
    # "TacSLTaskValve": TacSLTaskValve,
    # "TacSLTaskValveBig": TacSLTaskValveBig,
    "TacSLTaskBulb": TacSLTaskBulb,
}

class MultipleIsaacEnvWrapper():

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, cfg):
        self.cfg = cfg
        self.render_cache = []
        self._start_task(self.cfg)
        self.light_factor = self.cfg.light_factor

        # obtain task observation keys from the shape_meta
        shape_meta = self.cfg['shape_meta']
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        self.task_obs_keys = rgb_keys + lowdim_keys 
        # Exclude tactile keys from rgb_keys
        tactile_rgb_obs_keys = [
                                # 'tactile_force_field_left', \
                                # 'tactile_force_field_right', \
                                'tactile_depth_left', \
                                'tactile_depth_right', \
                                ]
        self.vision_obs_keys = [key for key in rgb_keys if key not in tactile_rgb_obs_keys]
        # add side view for better visualization
        if 'client' in self.task_obs_keys and 'client' not in self.vision_obs_keys:
            self.vision_obs_keys.append('client')
        # if 'wrist' not in self.vision_obs_keys:
        #     self.vision_obs_keys.append('wrist')
        # this case handles tactile-only setting
        if not self.vision_obs_keys:
            self.vision_obs_keys = ['front', 'side']

    @property
    def observation_space(self):
        obs_space = self.envs.observation_space
        obs_space = self._update_observation_space(obs_space)
        return obs_space
    
    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.envs.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.envs.num_observations

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.envs.num_environments

    def _start_task(self, cfg):
        rl_device = cfg.rl_device
        sim_device = cfg.sim_device
        graphics_device_id = cfg.graphics_device_id
        headless = cfg.headless
        virtual_screen_capture = cfg.capture_video
        force_render = cfg.force_render

        cfg_task = cfg['task']
        cfg_dict = omegaconf_to_dict(cfg_task)

        task_name = cfg_dict['name'] #e.g., TacSLTaskUSB

        self.envs = isaacgym_task_map[task_name](cfg=cfg_dict,
                                       rl_device=rl_device,
                                       sim_device=sim_device,
                                       graphics_device_id=graphics_device_id,
                                       headless=headless,
                                       virtual_screen_capture=virtual_screen_capture,
                                       force_render=force_render,
                                       )

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        # global rank of the GPU
        global_rank = 0
        # sets seed. if seed is -1 will pick a random one
        seed = set_seed(seed, torch_deterministic=self.cfg.torch_deterministic, rank=global_rank)
        # seed = set_seed(seed)
        self._seed = seed
        
    def step(self, actions):
        actions = torch.from_numpy(actions).to(dtype=torch.float32)
        obs, reward, reset, info = self.envs.step(actions)
        
        obs = self._transform_obs_data(obs['obs'])
        self.render_cache = obs
        obs = self._apply_obs_by_keys(obs)

        reset = reset.cpu().numpy()
        # reward = reward.cpu().numpy()
        reward = np.copy(reset)
        info = {key: value.cpu().numpy() for key, value in info.items() if isinstance(value, torch.Tensor)}

        return obs, reward, reset, info

    def reset(self):
        self.envs.reset_idx(torch.arange(self.num_envs))
        self.envs.compute_observations()
        obs = self.envs.reset()
        obs = self._transform_obs_data(obs['obs'])
        self.render_cache = obs
        obs = self._apply_obs_by_keys(obs)
        return obs
    

    def reset_to_dataset_state(self, **kwargs):
        return self.envs.reset_to_dataset_state(**kwargs)


    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img_list = []

        # If 'client' is available, compute the target size (height, width)
        target_shape = None
        for ref_key in ['client', 'front', 'wrist', 'side', 'wrist_2']:
            if ref_key in self.render_cache:
                target_shape = self.render_cache[ref_key].shape[2:4]  # (H, W)
                break

        for obs_view in self.vision_obs_keys:
            imgs = self.render_cache[obs_view]  # (num_envs, 3, H, W)
            # print(f"✅rendering view: {obs_view}, imgs shape: {imgs.shape}")
            
            if obs_view in ['tactile_force_field_left', 'tactile_force_field_right'] and target_shape is not None:
                resized_imgs = []
                for img in imgs:
                    # img: (3, 10, 14)
                    img = np.moveaxis(img, 0, -1) # (H, W, 3)
                    # print(f"✅shear_img shape before: {img.shape}")

                    # visualize tactile force field as RGB image
                    shear_img = visualize_tactile_shear_image(
                        img[..., 0],
                        img[..., 1:])
                    shear_img = (shear_img * 255).astype(np.uint8) # (H, W, 3)
                    shear_img = shear_img[:, :, ::-1] # swap BGR to RGB
                    shear_img = cv2.rotate(shear_img, cv2.ROTATE_90_CLOCKWISE)
                    shear_img = cv2.flip(shear_img, 1)
                    # print(f"✅shear_img shape after: {shear_img.shape}")

                    # cv2.resize expects size as (width, height)
                    resized_img = cv2.resize(shear_img, (target_shape[1], target_shape[0]))
                    resized_imgs.append(resized_img)
                imgs = np.stack(resized_imgs, axis=0)
                img_list.append(imgs)
            else:
                imgs = np.moveaxis(imgs, 1, -1)      # (num_envs, H, W, 3)
                imgs = (imgs * 255).astype(np.uint8)

                # For tactile camera views, if target_shape is defined, resize each image in the batch
                if obs_view in ['tactile_rgb_left', 'tactile_rgb_right'] and target_shape is not None:
                    resized_imgs = []
                    for img in imgs:
                        # cv2.resize expects size as (width, height)
                        resized_img = cv2.resize(img, (target_shape[1], target_shape[0]))
                        resized_imgs.append(resized_img)
                    imgs = np.stack(resized_imgs, axis=0)

                img_list.append(imgs)

        return np.concatenate(img_list, axis=2)


    def _transform_obs_data(self, obs):
        tf_obs = dict()
        for key, value in obs.items():
            # Squeeze singleton dimensions
            tf_obs[key] = value.cpu().numpy()

            # Check if the last dimension has size 3
            # check if it is the image data
            # (num_envs, H, W, 3) -> (num_envs, 3, H, W)
            if len(tf_obs[key].shape) >= 3 and tf_obs[key].shape[-1] == 3:
                tf_obs[key] = np.moveaxis(tf_obs[key], -1, 1)
                if key in ["front", "side", "wrist", "wrist_2", "client"]:
                    tf_obs[key] = self.light_condition(tf_obs[key], factor=self.light_factor)

        assert all(key in tf_obs for key in ('ee_pos', 'ee_quat')), \
                    "Keys 'ee_pos' and 'ee_quat' are missing in tf_obs"
        
        ee_pos = tf_obs['ee_pos']
        ee_quat = tf_obs['ee_quat']
        state = np.concatenate([ee_pos, ee_quat], axis=1)  # Shape: [2, 6]
        tf_obs['state'] = state

        return tf_obs

    def _apply_obs_by_keys(self, obs):
        task_obs = {key: obs[key] for key in self.task_obs_keys if key in obs}
        return task_obs

    def _update_observation_space(self, observation_space):
        updated_observation_space = {}
        for key, space in observation_space.spaces.items():
            if key in self.task_obs_keys: 
                if isinstance(space, spaces.Box):
                    # Check if the shape ends with (W, H, 3) and modify to (3, W, H)
                    if len(space.shape) == 3 and space.shape[-1] == 3:
                        updated_observation_space[key] = spaces.Box(
                            low=np.float32(-np.inf), high=np.float32(np.inf), 
                            shape=(3, space.shape[0], space.shape[1]), 
                            dtype=space.dtype
                        )
                    else:
                        # Keep the original space for non-image data
                        updated_observation_space[key] = space
                else:
                    # Keep the original space for non-Box types
                    updated_observation_space[key] = space

        if 'state' in self.task_obs_keys:
            updated_observation_space['state'] = spaces.Box(
                            low=np.float32(-np.inf), high=np.float32(np.inf), 
                            shape=(7,), 
                            dtype=np.float32
                        )

        # Return the updated Dict observation space
        return spaces.Dict(updated_observation_space)

    def light_condition(self, image, factor=0.5):
        if image.dtype in [np.float32, np.float64]:
            darkened = np.clip(image * factor, 0, 1)
        else:        
            darkened = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return darkened

    def close(self):
        pass
        # return self.envs.close()

def get_isaacgym_env(cfg):
    rl_device = cfg.rl_device
    sim_device = cfg.sim_device
    graphics_device_id = cfg.graphics_device_id
    headless = cfg.headless
    virtual_screen_capture = cfg.capture_video
    force_render = cfg.force_render

    cfg_task = cfg['task']
    cfg_dict = omegaconf_to_dict(cfg_task)

    env = TacSLTaskInsertion(cfg=cfg_dict,
                            rl_device=rl_device,
                            sim_device=sim_device,
                            graphics_device_id=graphics_device_id,
                            headless=headless,
                            virtual_screen_capture=virtual_screen_capture,
                            force_render=force_render,
                                    )
    return env

def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed