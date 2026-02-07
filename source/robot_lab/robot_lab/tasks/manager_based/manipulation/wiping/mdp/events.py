from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos

def reset_robot_joints(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """重置机器人关节到初始姿态，并加入微小噪声。"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取默认位置
    default_joint_pos = asset.data.default_joint_pos[env_ids]
    # 加入正负 0.05 弧度的随机噪声
    noise = sample_uniform(-0.05, 0.05, default_joint_pos.shape, device=env.device)
    initial_pos = default_joint_pos + noise
    initial_vel = torch.zeros_like(initial_pos)

    # 写入仿真
    asset.write_joint_state_to_sim(initial_pos, initial_vel, env_ids=env_ids)

def randomize_object_pose(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """随机化擦拭目标（如桌上的碗）的位置，让机器人学会自适应。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 生成随机位置 (x: 0.4~0.6, y: -0.1~0.1)
    new_pos = asset.data.default_root_state[env_ids, :3].clone()
    new_pos[:, 0] = sample_uniform(0.4, 0.6, (len(env_ids),), device=env.device)
    new_pos[:, 1] = sample_uniform(-0.1, 0.1, (len(env_ids),), device=env.device)
    
    # 保持原来的姿态和速度
    new_state = asset.data.default_root_state[env_ids].clone()
    new_state[:, :3] = new_pos
    
    # 写入仿真
    asset.write_root_state_to_sim(new_state, env_ids=env_ids)