from __future__ import annotations

import numpy as np
import torch

from .aviation_base_env import AviationBaseEnv
from .tello_map_search_env_cfg import TelloMapSearchEnvCfg


class TelloMapSeachEnv(AviationBaseEnv):
    cfg: TelloMapSearchEnvCfg
    def __init__(self, cfg:TelloMapSearchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


    def _pre_physics_step(self, actions) -> None:
        super()._pre_physics_step(actions)

    def _apply_action(self) -> None:
        super()._apply_action()

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        super()._get_dones()

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        super()._get_rewards()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        super()._get_observations()
    
    def _get_states(self) -> torch.Tensor:
        super()._get_states()

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)

    def _compute_intermediate_values(self) -> None:
        pass