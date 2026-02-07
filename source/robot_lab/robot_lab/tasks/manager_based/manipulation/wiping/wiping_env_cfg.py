# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.manipulation.wiping.mdp as mdp
from robot_lab.assets.booster_t1 import BOOSTER_T1_CFG 

# 定义你的本地资产基础路径
DATA_DIR = "/home/user/robot_lab/source/robot_lab/data/Props"

@configclass
class WipingSceneCfg(InteractiveSceneCfg):
    """配置包含真实桌子和苹果的交互场景"""
    
    # 1. 基础环境
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 2. 机器人：Booster T1
    robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 3. 桌子资产 (使用你找到的 table.usd)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(DATA_DIR, "table/table.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)), # 放在机器人前方
    )

    # 4. 目标物体：苹果 (使用你找到的 apple.usd)
    # 苹果的非规则表面将通过 MotorState.tau_est 反馈给机器人
    apple = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(DATA_DIR, "fruits/apple.usd"),
            # 开启 SDF 碰撞以获得平滑的擦拭体验
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                collision_approximation="sdf" 
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.75)), # 假设桌面高度 0.75m
    )

    # 5. 接触力传感器：挂载在末端
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_Hand_Roll", 
        history_length=3
    )

##
# MDP 组装 (对齐机器人接口文档)
##

@configclass
class ActionsCfg:
    """变阻抗动作：7维输入，包含 Kp 刚度控制"""
    arm_action = mdp.WipingVariableImpedanceActionCfg(
        asset_name="robot",
        pose_limit=0.02, # 限制每帧位移，模拟打磨质感
        nominal_kp=40.0, # 对应 MotorCmd.kp 标称值
    )

@configclass
class ObservationsCfg:
    """观测组：仅使用 LowState 中可用的数据"""
    @configclass
    class PolicyCfg(ObsGroup):
        # 电机反馈：q, dq, tau_est
        joint_pos = ObsTerm(func=mdp.joint_pos_raw)
        joint_vel = ObsTerm(func=mdp.joint_vel_raw)
        joint_tau = ObsTerm(func=mdp.joint_tau_est_raw)
        
        # 任务目标
        rel_ee_error = ObsTerm(
            func=mdp.ee_position_error_b, 
            params={"command_name": "wiping_cmd", "robot_cfg": SceneEntityCfg("robot", body_names=".*_Hand_Roll")}
        )
        
        # 惯性导航
        imu_rpy = ObsTerm(func=mdp.robot_imu_rpy)
        
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True # 开启噪声模拟 Sim2Real
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class WipingEnvCfg(ManagerBasedRLEnvCfg):
    """主环境配置"""
    scene: WipingSceneCfg = WipingSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: mdp.WipingCommandCfg = mdp.WipingCommandCfg() # 轨迹生成器
    
    # 奖励与终止逻辑
    rewards: mdp.RewardsCfg = mdp.RewardsCfg()
    terminations: mdp.TerminationsCfg = mdp.TerminationsCfg()
    events: mdp.EventCfg = mdp.EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005 # 200Hz 物理频率
        self.viewer.eye = (1.2, 1.2, 1.2)