
from __future__ import annotations

from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .aviation_base_env_cfg import AviationBaseEnvCfg


@configclass
class TelloMapSearchEnvCfg(AviationBaseEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 3.0
    
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # action and observation space
    def __post_init__(self):
        super().__post_init__()
        for agent_name in self.possible_agents:
            self.action_spaces[agent_name] = 2
            self.observation_spaces[agent_name] = 14
