from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class WipingCommand(CommandTerm):
    """针对擦拭任务的动态指令：在桌面上生成一个圆形或随机的目标点。"""
    cfg: WipingCommandCfg

    def __init__(self, cfg: WipingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 状态变量
        # targets: [x, y, z, qw, qx, qy, qz] (N, 7)
        self.targets = torch.zeros(self.num_envs, 7, device=self.device)
        # 轨迹进度索引
        self.steps = torch.zeros(self.num_envs, device=self.device)
        # 擦拭中心点 (每个环境可以随机在桌面的不同位置)
        self.centers = torch.zeros(self.num_envs, 3, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor):
        """当环境重置时，随机确定本次擦拭的中心点位置。"""
        # 在配置的范围内随机选择一个圆心
        r = self.cfg.center_range
        self.centers[env_ids, 0] = sample_uniform(r["x"][0], r["x"][1], (len(env_ids),), device=self.device)
        self.centers[env_ids, 1] = sample_uniform(r["y"][0], r["y"][1], (len(env_ids),), device=self.device)
        self.centers[env_ids, 2] = self.cfg.table_height  # 固定在桌面高度
        
        # 重置每个环境的进度
        self.steps[env_ids] = 0.0

    def _update_command(self):
        """每一帧更新目标点的位置，使其沿圆形轨迹移动。"""
        # 增加时间步 (dt * 速度系数)
        self.steps += self._env.step_dt * self.cfg.wiping_speed
        
        # 计算当前弧度
        theta = self.steps
        radius = self.cfg.wiping_radius
        
        # 计算圆形轨迹坐标
        # x = x_center + R * cos(theta)
        # y = y_center + R * sin(theta)
        self.targets[:, 0] = self.centers[:, 0] + radius * torch.cos(theta)
        self.targets[:, 1] = self.centers[:, 1] + radius * torch.sin(theta)
        self.targets[:, 2] = self.centers[:, 2] # 维持高度
        
        # 设置姿态：通常手掌垂直向下 (Euler: 0, 180°, 0)
        # 也可以根据 theta 动态调整手腕旋转
        palm_down_quat = quat_from_euler_xyz(
            torch.tensor(0.0, device=self.device),
            torch.tensor(3.14159, device=self.device), # 180度翻转，手掌朝下
            torch.tensor(0.0, device=self.device)
        ).repeat(self.num_envs, 1)
        
        self.targets[:, 3:7] = palm_down_quat

    @property
    def command(self) -> torch.Tensor:
        """返回给 Observation Manager 的指令向量。"""
        return self.targets

@configclass
class WipingCommandCfg(CommandTermCfg):
    """擦拭任务指令的配置参数。"""
    class_type: type = WipingCommand
    
    # 场景参数
    table_height: float = 0.75  # USD中桌面的Z轴高度
    
    # 轨迹参数
    wiping_radius: float = 0.1  # 擦拭圆圈的半径 (米)
    wiping_speed: float = 2.0   # 擦拭速度 (角速度)
    
    # 中心点随机范围 (x, y)
    center_range: dict[str, tuple[float, float]] = {
        "x": (0.4, 0.6),        # 机器人正前方 40cm 到 60cm
        "y": (-0.2, 0.2),       # 左右 20cm 范围
    }