from __future__ import annotations

import numpy as np
import torch
from abc import abstractmethod

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from .aviation_base_env_cfg import AviationBaseEnvCfg

class AviationBaseEnv(DirectMARLEnv):
    cfg: AviationBaseEnvCfg

    def __init__(self, cfg: AviationBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total env ids
        self.total_env_ids = torch.arange(self.num_envs, device=self.device)

        # Variables for Multi-Agent
        self._robots: dict[str, Articulation] = {}
        self.agent_key: list[str] = self.cfg.possible_agents
        self.num_agents = self.cfg.num_drones
    

    def _setup_scene(self):
        for i, robot_cfg in enumerate(self.cfg.robots):
            self._robots[f"robot_{i+1}"] = Articulation(robot_cfg)
            self.scene.articulations[f"robot_{i+1}"] = self._robots[f"robot_{i+1}"]

        # add ground
        spawn_ground_plane(prim_path=self.cfg.plane.prim_path, cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    
    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robots["robot_1"]._ALL_INDICES

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, 
                                                         high=int(self.max_episode_length))
        # ============ Robot State & Scene 리셋 ===============
        super()._reset_idx(env_ids)
        
        for robot in self._robots.values():
            pos_noise = sample_uniform(-0.125, 0.125, (len(env_ids, 3)), device=self.device)
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()
            robot_state = robot.data.default_root_state[env_ids].clone()
            robot_state[:, :3] += (pos_noise[:, :3] + self.scene.env_origins[env_ids])
            
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            robot.write_root_state_to_sim(robot_state, env_ids=env_ids)


    # ======================== Task-Specific Abstract Functions ==============================
    @abstractmethod
    def _pre_physics_step(self, actions):
        raise NotImplementedError(f"Please implement the '_pre_physics_step' method for {self.__class__.__name__}.")

    @abstractmethod
    def _apply_action(self):
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self):
        raise NotImplementedError(f"Please implement the '_get_done' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_rewards(self):
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_observations(self):
        raise NotImplementedError(f"Please implement the '_get_observation' method for {self.__class__.__name__}.")