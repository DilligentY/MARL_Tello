
from __future__ import annotations

from isaaclab.utils import configclass
from dataclasses import field
import copy

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab_assets import CRAZYFLIE_CFG

@configclass
class AviationBaseEnvCfg(DirectMARLEnvCfg):
    # env
    num_drones: int = 3
    episode_length_s: int
    decimation: int
    state_space: int
    possible_agents: list[str]
    action_space: dict[str, int]
    observation_space: dict[str, int]

    # ground plane
    plane: AssetBaseCfg =  AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
            spawn=GroundPlaneCfg(),
        )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # drone template
    drone_template: ArticulationCfg = CRAZYFLIE_CFG.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={".*": 0.0},
            joint_vel={
                "m1_joint":  200.0,
                "m2_joint": -200.0,
                "m3_joint":  200.0,
                "m4_joint": -200.0,
            },
        ),
    )

    # Drone list
    robots: list[ArticulationCfg] = field(default_factory=list)

    # Runtime시, Drone Configuration 생성
    def __post_init__(self):
        super().__post_init__()

        # 배치 간격(예: ±0.5 m씩 가로로 늘어놓기)
        spacing = 0.5
        origin_x = -(self.num_drones - 1) * spacing / 2.0

        self.robots = []
        for i in range(self.num_drones):
            cfg_i = copy.deepcopy(self.drone_template)
            cfg_i.prim_path = f"/World/envs/env_.*/Robot_{i+1}"
            cfg_i.init_state.pos = (
                origin_x + i * spacing,  # x 좌표
                0.0,                     # y 좌표
                0.5,                     # z 좌표
            )

            # 3) 리스트에 추가
            self.robots.append(cfg_i)
            self.possible_agents.append(f"robot_{i+1}")

    # # Drone 1
    # robot_1: ArticulationCfg = CRAZYFLIE_CFG.replace(
    #     prim_path="/World/envs/env_.*/Robot_1",
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=True,
    #         max_depenetration_velocity=10.0,
    #         enable_gyroscopic_forces=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(-0.5, 0.0, 0.5),
    #     joint_pos={
    #         ".*": 0.0,
    #     },
    #     joint_vel={
    #         "m1_joint": 200.0,
    #         "m2_joint": -200.0,
    #         "m3_joint": 200.0,
    #         "m4_joint": -200.0,
    #     },
    # ),
    # )

    # # Drone 2
    # robot_2: ArticulationCfg = CRAZYFLIE_CFG.replace(
    #     prim_path="/World/envs/env_.*/Robot_2",
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=True,
    #         max_depenetration_velocity=10.0,
    #         enable_gyroscopic_forces=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.5),
    #     joint_pos={
    #         ".*": 0.0,
    #     },
    #     joint_vel={
    #         "m1_joint": 200.0,
    #         "m2_joint": -200.0,
    #         "m3_joint": 200.0,
    #         "m4_joint": -200.0,
    #     },
    # ),
    # )

    # # Drone 3
    # robot_3: ArticulationCfg = CRAZYFLIE_CFG.replace(
    #     prim_path="/World/envs/env_.*/Robot_3",
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=True,
    #         max_depenetration_velocity=10.0,
    #         enable_gyroscopic_forces=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.5, 0.0, 0.5),
    #     joint_pos={
    #         ".*": 0.0,
    #         },
    #     joint_vel={
    #         "m1_joint": 200.0,
    #         "m2_joint": -200.0,
    #         "m3_joint": 200.0,
    #         "m4_joint": -200.0,
    #         },
    #     ),
    # )