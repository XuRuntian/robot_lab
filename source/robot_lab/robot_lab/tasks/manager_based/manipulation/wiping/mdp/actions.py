# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class WipingVariableImpedanceAction(ActionTerm):
    """
    【对齐官方例程版】自定义变阻抗动作项：
    1. 引入坐标系转换，确保 dx 指令与雅可比矩阵坐标系对齐。
    2. 增加 reset 同步逻辑，防止环境重置时发生物理爆炸。
    3. 完善 DLS-IK 数值稳定性。
    """
    
    cfg: WipingVariableImpedanceActionCfg
    _robot: Articulation

    def __init__(self, cfg: WipingVariableImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot = self._env.scene[cfg.asset_name]
        
        # 1. 索引解析
        self._ee_id = self._robot.find_bodies(cfg.body_name)[0][0]
        self._joint_ids, _ = self._robot.find_joints(cfg.joint_names)
        self._num_controlled_joints = len(self._joint_ids)
        
        # 2. 缓存关节限位 (用于积分截断)
        self._joint_limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids]

        # 3. Buffer 初始化
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        
        num_total_joints = self._robot.num_joints
        self.joint_pos_target = torch.zeros(self.num_envs, num_total_joints, device=self.device)
        self.joint_vel_target = torch.zeros_like(self.joint_pos_target)
        self.joint_kp = torch.zeros_like(self.joint_pos_target)
        self.joint_kd = torch.zeros_like(self.joint_pos_target)

        # 初始同步
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

    def reset(self, env_ids: torch.Tensor | None = None):
        """参考官方例程：重置时同步目标位置到当前真实位置"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # 关键：防止重置后 joint_pos_target 还留在上一轮的末尾导致瞬移
        self.joint_pos_target[env_ids] = self._robot.data.joint_pos[env_ids].clone()

    def _compute_dls_ik(self, jacobian, delta_pose, lambda_val=0.01):
        """阻尼最小二乘公式：dq = J^T * (J * J^T + lambda^2 * I)^-1 * dx"""
        B, _, N = jacobian.shape
        j_t = jacobian.transpose(1, 2)
        jj_t = torch.bmm(jacobian, j_t)
        damping = (lambda_val ** 2) * torch.eye(6, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        matrix = jj_t + damping
        # 求解线性方程组
        delta_q = torch.bmm(j_t, torch.linalg.solve(matrix, delta_pose.unsqueeze(-1))).squeeze(-1)
        return delta_q

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        
        # 1. 获取雅可比并处理固定基座偏移 (参考 run_diff_ik.py 第 127 行逻辑)
        all_jacobians = self._robot.root_physx_view.get_jacobians()
        physx_link_count = all_jacobians.shape[1]
        actual_ee_id = self._ee_id - 1 if self._robot.is_fixed_base else self._ee_id
        ik_jacobians = all_jacobians[:, actual_ee_id, :, :][:, :, self._joint_ids]

        # 2. 坐标系转换：核心修正
        # 将世界坐标系下的动作转换到机器人局部坐标系 (Root Frame)
        pose_actions = torch.clamp(actions[:, :6], -1.0, 1.0)
        v_world = pose_actions[:, :3] * self.cfg.scale
        w_world = pose_actions[:, 3:6] * self.cfg.scale * 2.0
        
        # 获取机器人当前的朝向
        root_quat = self._robot.data.root_quat_w
        # 旋转指令向量：dx_local = R_inv * dx_world
        v_local = math_utils.quat_rotate_inverse(root_quat, v_world)
        w_local = math_utils.quat_rotate_inverse(root_quat, w_world)
        command_local = torch.cat([v_local, w_local], dim=1)

        # 3. 计算并积分
        delta_q = self._compute_dls_ik(ik_jacobians, command_local)
        self.joint_pos_target[:, self._joint_ids] += delta_q

        # 4. 边界截断与 PD 参数更新
        self.joint_pos_target[:, self._joint_ids] = torch.clamp(
            self.joint_pos_target[:, self._joint_ids], 
            self._joint_limits[..., 0], 
            self._joint_limits[..., 1]
        )

        stiffness_actions = torch.clamp(actions[:, 6], 0.1, 5.0)
        target_kp = self.cfg.nominal_kp * stiffness_actions.unsqueeze(1)
        self.joint_kp[:, self._joint_ids] = target_kp
        self.joint_kd[:, self._joint_ids] = 2.0 * torch.sqrt(target_kp) * self.cfg.damping_ratio

        # --- 调试打印 (每 50 步) ---
        if self._env.common_step_counter % 50 == 0:
            ee_name = self._robot.data.body_names[self._ee_id]
            print(f"\n[IK Sync] EE: {ee_name} | DX_Local Norm: {torch.norm(v_local).item():.4f}")
            print(f"DQ Max: {torch.max(torch.abs(delta_q)).item():.6f}")

    def apply_actions(self):
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
    scale: float = 0.2  
    nominal_kp: float = 800.0 
    damping_ratio: float = 1.0