# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class WipingVariableImpedanceAction(ActionTerm):
    """自定义变阻抗动作项：实现 EE 位姿增量控制 + 动态刚度映射 (完全对齐 MotorCmd)"""
    
    cfg: WipingVariableImpedanceActionCfg
    _robot: Articulation

    def __init__(self, cfg: WipingVariableImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # 获取机器人引用
        self._robot = self._env.scene[cfg.asset_name]
        
        # 内部 Buffer 初始化
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        
        # 预分配对齐 MotorCmd 的关节指令 Buffer
        # 注意：这里假设只控制手臂关节，实际应用中建议通过 cfg.joint_ids 过滤
        num_joints = self._robot.num_joints
        self.joint_pos_target = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.joint_vel_target = torch.zeros_like(self.joint_pos_target)
        self.joint_kp = torch.zeros_like(self.joint_pos_target)
        self.joint_kd = torch.zeros_like(self.joint_pos_target)
        self.joint_tau_ff = torch.zeros_like(self.joint_pos_target)

    """
    Properties (必须实现的抽象属性)
    """

    @property
    def action_dim(self) -> int:
        # 6 (EE 增量) + 1 (Kp 缩放)
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        """返回 RL 输出的原始动作"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """返回处理后的动作"""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """处理 RL 输出，将其转化为内部指令"""
        self._raw_actions[:] = actions
        
        # 1. 裁剪并赋值给已处理动作
        self._processed_actions[:, :6] = torch.clamp(actions[:, :6], -1.0, 1.0)
        self._processed_actions[:, 6] = torch.clamp(actions[:, 6], 0.1, 2.0) 

        # 2. 提取参数
        ee_delta_pose = self._processed_actions[:, :6] * self.cfg.pose_limit
        stiffness_scale = self._processed_actions[:, 6:7]

        # --- 核心修正：获取当前环境下的关节总数 ---
        num_joints = self._robot.num_joints 
        current_joint_pos = self._robot.data.joint_pos

        # 3. 更新关节位置目标
        # 修正逻辑：确保 repeat 的维度和切片正确
        repeat_count = (num_joints // 6) + 1
        delta_expanded = ee_delta_pose.repeat(1, repeat_count)[:, :num_joints]
        self.joint_pos_target[:] = current_joint_pos + (delta_expanded * 0.1)
        
        # 4. 动态映射 Kp 与 Kd
        self.joint_kp[:] = self.cfg.nominal_kp * stiffness_scale
        self.joint_kd[:] = 2.0 * torch.sqrt(self.joint_kp) * self.cfg.damping_ratio

    def apply_actions(self):
        """将计算结果通过低级接口下发给物理引擎"""
        # 对应真机 rt/joint_ctrl 接口的数据下发
    
    # 1. 设置关节刚度 (Kp) 和 阻尼 (Kd)
        self._robot.write_joint_stiffness_to_sim(self.joint_kp)
        self._robot.write_joint_damping_to_sim(self.joint_kd)
        
        # 2. 设置关节目标位置 (q) 和 目标速度 (v)
        self._robot.set_joint_position_target(self.joint_pos_target)
        self._robot.set_joint_velocity_target(self.joint_vel_target)
        
        # 3. 设置力矩前馈项 (tau_ff)
        self._robot.set_joint_effort_target(self.joint_tau_ff)

@configclass
class WipingVariableImpedanceActionCfg(ActionTermCfg):
    """变阻抗动作项的配置类"""
    class_type: type = WipingVariableImpedanceAction
    asset_name: str = "robot"
    pose_limit: float = 0.05      
    nominal_kp: float = 40.0     
    damping_ratio: float = 1.0