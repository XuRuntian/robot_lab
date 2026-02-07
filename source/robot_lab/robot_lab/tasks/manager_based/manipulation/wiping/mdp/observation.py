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
    # 仿真中使用 joint_force 模拟真机的估算力矩
    return asset.data.joint_force[:, asset_cfg.joint_ids]

def robot_imu_rpy(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取基座 IMU 欧拉角 (对应 ImuState.rpy)"""
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, rpy = euler_xyz_from_quat(asset.data.root_quat_w)
    return rpy

def robot_imu_gyro(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取基座角速度 (对应 ImuState.gyro)"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w

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