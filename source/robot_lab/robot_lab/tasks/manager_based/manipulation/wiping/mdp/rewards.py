from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- 追踪奖励 (参考你提供的 exp 风格) ---

def ee_position_tracking_exp(env: ManagerBasedRLEnv, std: float, command_name: str, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """末端位置追踪奖励：鼓励 EE 靠近目标点"""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[robot_cfg.name]
    
    # 计算末端当前位置与目标位置的欧氏距离平方
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    target_pos_w = command.targets[:, :3]
    error = torch.sum(torch.square(target_pos_w - ee_pos_w), dim=-1)
    
    return torch.exp(-error / std**2)

def ee_orientation_tracking_exp(env: ManagerBasedRLEnv, std: float, command_name: str, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """末端姿态追踪奖励：鼓励手掌始终向下"""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[robot_cfg.name]
    
    ee_quat_w = robot.data.body_quat_w[:, robot_cfg.body_ids[0]]
    target_quat_w = command.targets[:, 3:7]
    error = quat_error_magnitude(ee_quat_w, target_quat_w) ** 2
    
    return torch.exp(-error / std**2)

# --- 力控与交互奖励 (针对打磨任务新增) ---

def constant_force_tracking_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    target_force: float, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """恒力追踪奖励：对应真机 MotorState.tau_est 产生的法向力反馈"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取末端法向力 (假设 Z 轴为法向)
    current_force = sensor.data.net_forces_w[:, sensor_cfg.body_ids[0], 2]
    
    # 计算实际力与目标力（如 5N）的偏差
    error = torch.square(current_force - target_force)
    
    return torch.exp(-error / std**2)

def ee_velocity_smoothing(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚末端抖动：打磨任务需要平滑的切向运动"""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_vel_w = robot.data.body_lin_vel_w[:, robot_cfg.body_ids[0]]
    
    # 惩罚过大的线速度平方，促使运动平顺
    return -torch.sum(torch.square(ee_vel_w), dim=-1)

# --- 惩罚项 ---

def joint_torque_penalty(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚过大的关节力矩：保护真机电机"""
    robot: Articulation = env.scene[robot_cfg.name]
    # 对应 MotorCmd.tau 限制
    return -torch.sum(torch.square(robot.data.applied_torque), dim=-1)