from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class WipingVariableImpedanceAction(ActionTerm):
    """自定义动作项：实现位姿跟踪 + 动态刚度调整 (对齐 MotorCmd 接口)"""
    
    cfg: WipingVariableImpedanceActionCfg
    _robot: Articulation

    def __init__(self, cfg: WipingVariableImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot = self._env.scene[cfg.asset_name]
        
        # 预分配指令张量 (对应 MotorCmd 的核心参数)
        self.joint_pos_target = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        self.joint_vel_target = torch.zeros_like(self.joint_pos_target)
        self.joint_kp = torch.zeros_like(self.joint_pos_target)
        self.joint_kd = torch.zeros_like(self.joint_pos_target)
        self.joint_tau_ff = torch.zeros_like(self.joint_pos_target)

    @property
    def action_dim(self) -> int:
        # 6 (EE位姿增量: dx, dy, dz, droll, dpitch, dyaw) + 1 (刚度缩放 K_scale)
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions
        
        # 1. 提取 RL 输出的位姿增量与刚度缩放
        ee_delta_pose = actions[:, :6] * self.cfg.pose_limit
        stiffness_scale = torch.clamp(actions[:, 6:7], 0.1, 2.0) # 映射到 [0.1, 2.0] 倍率

        # 2. 这里的核心逻辑：将 EE 增量转化为目标关节位置 q (模拟串联结构控制)
        # 注意：此处简化了 IK 过程，实际通常配合 DifferentialInverseKinematics 使用
        # ... (IK 计算逻辑) ...
        
        # 3. 动态映射 Kp 与 Kd (这是对齐真机 MotorCmd 的关键)
        # 将 RL/EMG 驱动的缩放系数作用于标称刚度
        self.joint_kp[:] = self.cfg.nominal_kp * stiffness_scale
        # 保持临界阻尼比计算 Kd，确保不发生震荡
        self.joint_kd[:] = 2.0 * torch.sqrt(self.joint_kp) * self.cfg.damping_ratio

    def apply_actions(self):
        """将处理后的 MotorCmd 数据写入物理引擎"""
        # 设置 PD 增益与目标值
        self._robot.set_joint_stiffness_target(self.joint_kp)
        self._robot.set_joint_damping_target(self.joint_kd)
        self._robot.set_joint_position_target(self.joint_pos_target)
        self._robot.set_joint_velocity_target(self.joint_vel_target)
        # 设置力矩前馈项 tau
        self._robot.set_joint_effort_target(self.joint_tau_ff)

@configclass
class WipingVariableImpedanceActionCfg(ActionTermCfg):
    class_type: type = WipingVariableImpedanceAction
    asset_name: str = "robot"
    pose_limit: float = 0.05      # 每一帧最大位移限制
    nominal_kp: float = 40.0     # 标称刚度 (根据机器人文档设置)
    damping_ratio: float = 1.0   # 阻尼比 (1.0 代表临界阻尼)