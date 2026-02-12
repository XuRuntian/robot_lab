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
    """
    【完全修复版】自定义变阻抗动作项：
    1. 采用积分式 IK 更新，消除运动滞后。
    2. 自动适配 PhysX 基座偏移。
    3. 增加关节限位保护。
    """
    
    cfg: WipingVariableImpedanceActionCfg
    _robot: Articulation

    def __init__(self, cfg: WipingVariableImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot = self._env.scene[cfg.asset_name]
        
        # 1. 解析索引
        self._ee_id = self._robot.find_bodies(cfg.body_name)[0][0]
        self._joint_ids, _ = self._robot.find_joints(cfg.joint_names)
        self._num_controlled_joints = len(self._joint_ids)
        
        # 2. 缓存关节限位 (用于积分截断)
        # 形状: [num_envs, num_controlled_joints, 2]
        self._joint_limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids]

        # 3. Buffer 初始化
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        
        num_total_joints = self._robot.num_joints
        self.joint_pos_target = torch.zeros(self.num_envs, num_total_joints, device=self.device)
        self.joint_vel_target = torch.zeros_like(self.joint_pos_target)
        self.joint_kp = torch.zeros_like(self.joint_pos_target)
        self.joint_kd = torch.zeros_like(self.joint_pos_target)

        # 初始位置设为当前默认姿态
        self.joint_pos_target[:] = self._robot.data.default_joint_pos.clone()

    @property
    def action_dim(self) -> int:
        return 7 

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _compute_dls_ik(self, jacobian, delta_pose, lambda_val=1e-3):
        """阻尼最小二乘逆解，lambda 调小以增强小位移响应"""
        B, _, N = jacobian.shape
        j_t = jacobian.transpose(1, 2)
        jj_t = torch.bmm(jacobian, j_t)
        # 单位阵适配
        damping = (lambda_val ** 2) * torch.eye(6, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        rhs = delta_pose.unsqueeze(-1)
        matrix = jj_t + damping
        # 解方程 (J*J^T + lambda^2*I) * x = dx
        intermediate_result = torch.linalg.solve(matrix, rhs)
        delta_q = torch.bmm(j_t, intermediate_result).squeeze(-1)
        return delta_q

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        
        # 动作截断与解析
        pose_actions = torch.clamp(actions[:, :6], -1.0, 1.0)
        stiffness_actions = torch.clamp(actions[:, 6], 0.1, 5.0)
        self._processed_actions[:, :6] = pose_actions
        self._processed_actions[:, 6] = stiffness_actions

        # 1. 获取雅可比并处理偏移
        all_jacobians = self._robot.root_physx_view.get_jacobians()
        physx_link_count = all_jacobians.shape[1]
        actual_ee_id = self._ee_id - 1 if physx_link_count < self._robot.num_bodies else self._ee_id
        ik_jacobians = all_jacobians[:, actual_ee_id, :, :][:, :, self._joint_ids]

        # 2. 构造末端指令 dx (加上 scale)
        # 注意：scale 在 Cfg 中设为 0.5 左右比较合适
        command = torch.cat([
            pose_actions[:, :3] * self.cfg.scale,       
            pose_actions[:, 3:6] * self.cfg.scale * 2.0 
        ], dim=1)

        # 3. 计算关节空间增量 dq
        delta_q = self._compute_dls_ik(ik_jacobians, command)

        # 4. 【核心修复】积分更新：在现有目标上累加，而不是在实际位置上累加
        # 这样即使机器人一步没跟上，目标点也会持续拉动，产生更大的误差力矩
        self.joint_pos_target[:, self._joint_ids] += delta_q

        # 5. 安全检查：强制关节限位截断，防止“飞出”
        self.joint_pos_target[:, self._joint_ids] = torch.clamp(
            self.joint_pos_target[:, self._joint_ids], 
            self._joint_limits[..., 0], 
            self._joint_limits[..., 1]
        )

        # 6. 更新变阻抗参数 (PD Gain)
        target_kp = self.cfg.nominal_kp * stiffness_actions.unsqueeze(1)
        target_kd = 2.0 * torch.sqrt(target_kp) * self.cfg.damping_ratio
        
        self.joint_kp[:, self._joint_ids] = target_kp
        self.joint_kd[:, self._joint_ids] = target_kd

        # 调试打印 (建议每 50 步打印一次确认运动强度)
        if self._env.common_step_counter % 50 == 0:
            print(f"--- Action Loop ---")
            print(f"DQ Max: {torch.max(torch.abs(delta_q)).item():.6f}")
            print(f"Target Pos Sample: {self.joint_pos_target[0, self._joint_ids[0]].item():.4f}")

    def apply_actions(self):
        # 将指令下发至物理引擎
        self._robot.write_joint_stiffness_to_sim(self.joint_kp)
        self._robot.write_joint_damping_to_sim(self.joint_kd)
        self._robot.set_joint_position_target(self.joint_pos_target)
        self._robot.set_joint_velocity_target(self.joint_vel_target)

@configclass
class WipingVariableImpedanceActionCfg(ActionTermCfg):
    class_type: type = WipingVariableImpedanceAction
    asset_name: str = "robot"
    body_name: str = "right_Link22" 
    joint_names: list[str] = [".*Right.*", ".*right.*"]
    
    # 建议将 scale 保持在 0.1 ~ 0.5 之间
    # 如果策略输出是 1.0，则单步最大移动 0.1m 或 0.5m
    scale: float = 0.2  
    nominal_kp: float = 800.0 
    damping_ratio: float = 1.0