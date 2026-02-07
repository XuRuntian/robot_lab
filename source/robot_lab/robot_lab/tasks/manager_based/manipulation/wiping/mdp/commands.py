# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class WipingCommand(CommandTerm):
    """针对擦拭任务的动态指令。"""

    def __init__(self, cfg: WipingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 初始化张量
        self.targets = torch.zeros(self.num_envs, 7, device=self.device)
        self.theta = torch.zeros(self.num_envs, device=self.device)
        self.centers = torch.zeros(self.num_envs, 3, device=self.device)

    # --- 修正这里：名字必须叫 _resample_command ---
    def _resample_command(self, env_ids: torch.Tensor):
        r = self.cfg.center_range
        self.centers[env_ids, 0] = sample_uniform(r["x"][0], r["x"][1], (len(env_ids),), device=self.device)
        self.centers[env_ids, 1] = sample_uniform(r["y"][0], r["y"][1], (len(env_ids),), device=self.device)
        self.centers[env_ids, 2] = self.cfg.table_height
        # 随机化初始相位
        self.theta[env_ids] = sample_uniform(0, 6.28, (len(env_ids),), device=self.device)

    def _update_command(self):
        """每一帧更新目标点位置。"""
        self.theta += self._env.step_dt * self.cfg.wiping_speed
        radius = self.cfg.wiping_radius
        self.targets[:, 0] = self.centers[:, 0] + radius * torch.cos(self.theta)
        self.targets[:, 1] = self.centers[:, 1] + radius * torch.sin(self.theta)
        self.targets[:, 2] = self.centers[:, 2]
        
        # 手掌垂直向下
        palm_down_quat = quat_from_euler_xyz(
            torch.tensor(0.0, device=self.device),
            torch.tensor(3.14159, device=self.device),
            torch.tensor(0.0, device=self.device)
        )
        self.targets[:, 3:7] = palm_down_quat.repeat(self.num_envs, 1)

    def _update_metrics(self):
        """补全另一个抽象方法。"""
        return {}

    @property
    def command(self) -> torch.Tensor:
        return self.targets

@configclass
class WipingCommandCfg(CommandTermCfg):
    """擦拭指令配置。"""
    class_type: type = WipingCommand
    
    table_height: float = 0.73  # 对齐你 USD 的桌子高度
    wiping_radius: float = 0.1
    wiping_speed: float = 2.0   
    center_range: dict[str, tuple[float, float]] = {
        "x": (0.45, 0.55),
        "y": (-0.1, 0.1),
    }