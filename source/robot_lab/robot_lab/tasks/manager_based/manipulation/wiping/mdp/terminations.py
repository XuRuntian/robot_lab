# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.terminations import time_out
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- 核心终止逻辑 ---

def bad_ee_distance(env: ManagerBasedRLEnv, command_name: str, threshold: float, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """如果末端执行器（EE）距离目标点太远，视为任务失败"""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[robot_cfg.name]
    
    # 获取末端当前位置
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    target_pos_w = command.targets[:, :3]
    
    # 计算欧氏距离偏差
    distance = torch.norm(target_pos_w - ee_pos_w, dim=1)
    return distance > threshold

def bad_ee_orientation(env: ManagerBasedRLEnv, command_name: str, threshold: float, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """如果手掌姿态翻转严重（偏离垂直向下太远），则重置"""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[robot_cfg.name]
    
    ee_quat_w = robot.data.body_quat_w[:, robot_cfg.body_ids[0]]
    target_quat_w = command.targets[:, 3:7]
    
    # 计算角度偏差
    error = math_utils.quat_error_magnitude(ee_quat_w, target_quat_w)
    return error > threshold

def excessive_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """【力控专属】如果法向力超过阈值（可能压坏零件或真机过载），则强制停止"""
    # 对应 MotorState.tau_est 转换出的末端力反馈
    sensor = env.scene.sensors[sensor_cfg.name]
    net_force_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids[0], 2].abs()
    
    return net_force_z > threshold

def robot_fallen(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, z_threshold: float = 0.5) -> torch.Tensor:
    """如果机器人躯干高度低于阈值（摔倒），重置"""
    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_z = robot.data.root_pos_w[:, 2]
    return root_pos_z < z_threshold