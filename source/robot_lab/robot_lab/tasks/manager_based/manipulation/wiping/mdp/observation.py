from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# --- 机器人本体感知 (对齐 LowState 文档) ---

def joint_pos_raw(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取关节位置 (对应 MotorState.q)"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]

def joint_vel_raw(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取关节速度 (对应 MotorState.dq)"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]

def joint_tau_est_raw(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取关节估算力矩 (对应 MotorState.tau_est)"""
    asset: Articulation = env.scene[asset_cfg.name]
    # 仿真中使用 joint_tau 模拟真机的估算力矩
    return asset.data.computed_torque[:, asset_cfg.joint_ids]

def robot_imu_rpy(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取基座 IMU 欧拉角 (对应 ImuState.rpy)，返回 (num_envs, 3)"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 1. 获取三个轴的欧拉角，每个形状都是 (num_envs,)
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    
    # 2. 将它们在最后一个维度拼接起来，形成 (num_envs, 3)
    rpy = torch.stack([roll, pitch, yaw], dim=-1)
    
    # [DEBUG] 现在它应该显示类似 tensor([[0.0, 0.0, 0.0]], device='cuda:0')
    # print(f"[DEBUG] 机器人基座 IMU 欧拉角 (rpy) 形状: {rpy.shape}")
    
    return rpy

def robot_imu_gyro(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取基座角速度 (对应 ImuState.gyro)"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w

def last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """获取环境上一时刻执行的动作值，用于提升策略的连续性"""
    return env.action_manager.action
# --- 任务相关观测 (擦拭任务必需) ---

def ee_position_error_b(env: ManagerBasedEnv, command_name: str, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """末端相对于目标的位姿偏差（在机器人基座坐标系下）"""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[robot_cfg.name]
    
    # 获取末端当前位置与目标位置
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    target_pos_w = command.targets[:, :3]
    
    # 计算偏差并转换到基座坐标系
    root_quat_w = robot.data.root_quat_w
    return quat_apply(quat_inv(root_quat_w), target_pos_w - ee_pos_w)