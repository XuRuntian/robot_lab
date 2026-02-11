# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform, sample_log_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def _randomize_prop_by_op(
    current_values: torch.Tensor,
    distribution_params: tuple[float, float],
    env_ids: torch.Tensor,
    joint_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """内部辅助函数：执行具体的随机化数学运算。"""
    num_envs = len(env_ids)
    if isinstance(joint_ids, slice):
        num_joints = current_values.shape[1]
    else:
        num_joints = len(joint_ids)
    
    shape = (num_envs, num_joints)
    low, high = distribution_params

    if distribution == "uniform":
        noise = sample_uniform(low, high, shape, device=current_values.device)
    elif distribution == "log_uniform":
        noise = sample_log_uniform(low, high, shape, device=current_values.device)
    elif distribution == "gaussian":
        noise = torch.randn(shape, device=current_values.device) * high + low
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    new_values = current_values.clone()
    
    # 获取需要修改的索引视图
    if not isinstance(joint_ids, slice):
        indices = (env_ids[:, None], joint_ids)
    else:
        indices = (env_ids, joint_ids)

    if operation == "abs":
        new_values[indices] = noise
    elif operation == "add":
        new_values[indices] += noise
    elif operation == "scale":
        new_values[indices] *= noise

    return new_values

def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    随机化关节默认位置（模拟标定误差）。
    通常在 'startup' 阶段调用。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 1. 保护机制：确保我们有一个永远干净的“出厂设置” (nominal)
    if not hasattr(asset.data, "default_joint_pos_nominal"):
        asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=asset.device)

    # 2. 基于 nominal 值进行随机化，而不是基于上一次的 default 值 (防止漂移)
    # 注意：这里我们读取的是 nominal，应用到 default
    # 如果你想基于当前的 default 叠加，就把下面的 nominal 改回 default_joint_pos
    base_values = asset.data.default_joint_pos_nominal.unsqueeze(0).repeat(env.scene.num_envs, 1)

    if pos_distribution_params is not None:
        new_default_pos = _randomize_prop_by_op(
            base_values, # 使用干净的基准值
            pos_distribution_params, 
            env_ids, 
            joint_ids, 
            operation=operation, 
            distribution=distribution
        )
        
        # 3. 更新 asset 数据缓存 (这会影响 reset_robot_joints 的行为)
        if not isinstance(joint_ids, slice):
            asset.data.default_joint_pos[env_ids[:, None], joint_ids] = \
                new_default_pos[env_ids[:, None], joint_ids]
        else:
            asset.data.default_joint_pos[env_ids] = new_default_pos[env_ids]

        # 4. 同步更新 ActionManager 中的偏移量 (仅当确实需要改变控制零点时)
        # 警告：如果你只是想让机器人出生位置变，但不想改变控制器的目标零点，请注释掉这段
        if "arm_action" in env.action_manager._terms:
            try:
                action_term = env.action_manager.get_term("arm_action")
                if hasattr(action_term, "_offset"):
                    if not isinstance(joint_ids, slice):
                        action_term._offset[env_ids[:, None], joint_ids] = \
                            asset.data.default_joint_pos[env_ids[:, None], joint_ids]
                    else:
                        action_term._offset[env_ids] = asset.data.default_joint_pos[env_ids]
            except LookupError:
                pass # 如果没找到 arm_action 也不要报错崩溃

def reset_robot_joints(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """
    重置机器人关节到初始姿态。
    通常在 'reset' 阶段调用。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 1. 获取默认关节位置 (这个值可能已经被 randomize_joint_default_pos 修改过了，这是预期的)
    default_joint_pos = asset.data.default_joint_pos[env_ids]
    
    # 2. 添加复位噪声 (模拟每次回零的微小误差，例如 +/- 0.02 弧度)
    noise = sample_uniform(-0.02, 0.02, default_joint_pos.shape, device=env.device)
    
    # 3. 强制写入仿真器 (同时将速度重置为 0)
    asset.write_joint_state_to_sim(default_joint_pos + noise, torch.zeros_like(default_joint_pos), env_ids=env_ids)

def reset_object_pose(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """
    【核心修复】重置擦拭对象的位置，兼容 XFormPrim 和 RigidObject。
    已修复采样范围错误 (low > high)。
    """
    asset = env.scene[asset_cfg.name]
    
    # 1. 检查是否是 RigidObject (具有 .data 属性)
    if hasattr(asset, "data"):
        # 获取默认状态 [num_envs, 13]
        new_state = asset.data.default_root_state[env_ids].clone()
        
        # 2. 随机化 X/Y 坐标
        # 修正：low (0.55) 必须小于 high (0.65)
        new_state[:, 0] = sample_uniform(0.55, 0.65, (len(env_ids),), device=env.device)
        new_state[:, 1] = sample_uniform(-0.1, 0.1, (len(env_ids),), device=env.device)
        # Z 轴对齐桌面
        new_state[:, 2] = 0.76 
        
        # 3. 写入仿真 (RigidObject 专用)
        asset.write_root_state_to_sim(new_state, env_ids=env_ids)
    else:
        # 如果是 XFormPrim (没有 .data)
        pos = torch.zeros((len(env_ids), 3), device=env.device)
        
        # 修正：low (0.55) 必须小于 high (0.65)
        pos[:, 0] = sample_uniform(0.55, 0.65, (len(env_ids),), device=env.device)
        pos[:, 1] = sample_uniform(-0.1, 0.1, (len(env_ids),), device=env.device)
        pos[:, 2] = 0.76
        
        # 默认朝向 (单位四元数)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(env_ids), 1)
        
        # 4. 写入仿真 (XFormPrim 专用 API)
        asset.set_world_poses(pos, quat, indices=env_ids)