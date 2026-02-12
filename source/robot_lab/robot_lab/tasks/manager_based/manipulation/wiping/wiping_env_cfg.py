# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import MISSING

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
from isaaclab.assets import RigidObjectCfg 

import robot_lab.tasks.manager_based.manipulation.wiping.mdp as mdp
from robot_lab.assets.booster_gripper import BOOSTER_T1_CFG

# 资产基础路径
DATA_DIR = "/home/user/robot_lab/source/robot_lab/data/Props"

##
# Scene definition
##

@configclass
class WipingSceneCfg(InteractiveSceneCfg):
    """配置包含真实桌子和苹果的交互场景"""
    # 地面与灯光
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 机器人：Booster T1 (资产在 config 层注入)
    robot: ArticulationCfg = MISSING

    # 桌子资产
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(DATA_DIR, "table/desk.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            scale=(0.07, 0.07, 0.07),

        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.65, 0.0, 0.0), #0.65,0,0
            rot=(0.7071, 0.0, 0.0, 0.7071)       
        )
    )

    # 目标物体：苹果 (测试变阻抗的核心)
    # apple = RigidObjectCfg( # 修正：从 AssetBaseCfg 改为 RigidObjectCfg
    #     prim_path="{ENV_REGEX_NS}/TargetBlock",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(DATA_DIR, "fruits/apple.usd"),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False, 
    #             disable_gravity=False,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=True,
    #         ),
    #         scale=(0.01, 0.01, 0.01),
    #     ),
    #     # 此时这个 pos 拥有极高优先级，会强行覆盖 USD 默认值
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.85)), 
    # )
    apple = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetBlock",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.03),
            # 暴力破解 1：强制关闭实例化，解开空间锁
            func=sim_utils.spawn_mesh_cuboid, 
            # 强制设置物理
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 1.0)), # 修正：设置苹果初始位置
    )
    # 接触力传感器 (对应 LowState 需求)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_Link(11|22)", history_length=3, debug_vis=True
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """轨迹指令配置"""
    wiping = mdp.WipingCommandCfg() # 假设你在 mdp 里定义了这个类

@configclass
class ActionsCfg:
    """变阻抗动作配置"""
    # 7维：6D位姿 + 1D刚度缩放
    arm_action = mdp.WipingVariableImpedanceActionCfg(
        asset_name="robot",
        
        # 1. 确保末端 Link 名字正确 (看你之前的 Log 是 right_Link22)
        body_name="right_Link22",
        
        # 2. 确保关节正则正确
        joint_names=[".*Right.*", ".*right.*"],
        
        # 3. 参数调整
        scale=0.05,        # 每次移动 5cm
        nominal_kp=800.0,  # 基础刚度
        damping_ratio=1.0, # 阻尼比
        
        # ❌ 删除 controller=... 这一行！千万别留！
    )


@configclass
class ObservationsCfg:
    """观测组配置：对齐真机 LowState"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        # 核心修正：每一个获取关节数据的函数都需要指定 asset_cfg
        joint_pos = ObsTerm(
            func=mdp.joint_pos_raw, 
            params={"asset_cfg": SceneEntityCfg("robot")}, # 告诉它看哪台机器人
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_raw, 
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        joint_tau = ObsTerm(
            func=mdp.joint_tau_est_raw,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        # 任务目标 (你之前已经写了 params，保持即可)
        rel_ee_error = ObsTerm(
            func=mdp.ee_position_error_b, 
            params={
                "command_name": "wiping", 
                "robot_cfg": SceneEntityCfg("robot", body_names=".*_Link(11|22)")
            }
        )
        
        # 惯性导航
        imu_rpy = ObsTerm(
            func=mdp.robot_imu_rpy,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        # 历史动作 (通常不需要参数，因为它直接读 ActionManager)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """事件配置：随机化与重置"""
    # 模拟校准误差
    randomize_calibration = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "pos_distribution_params": (-0.01, 0.01), "operation": "add"}
    )

    reset_robot = EventTerm(
        func=mdp.reset_robot_joints, # 调用你在 mdp.py 里写的重置函数
        mode="reset",                # 在每次 env.reset() 时触发
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # 重置物体位置
    reset_apple = EventTerm(
        func=mdp.reset_object_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("apple")}
    )

@configclass
class RewardsCfg:
    """奖励项配置"""
    # 1. 追踪奖励 (指数型)
    pos_tracking = RewTerm(
        func=mdp.ee_position_tracking_exp, 
        weight=1.0, 
        params={"std": 0.2, "command_name": "wiping", "robot_cfg": SceneEntityCfg("robot", body_names=".*_Link(11|22)")}
    )
    # 2. 恒力奖励 (核心)
    force_tracking = RewTerm(
        func=mdp.constant_force_tracking_exp, 
        weight=5.0, 
        params={"std": 2.0, "target_force": 5.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Link(11|22)")}
    )
    # 3. 功耗惩罚
    torque_penalty = RewTerm(
        func=mdp.joint_torque_penalty, 
        weight=-1e-4,
        params={"robot_cfg": SceneEntityCfg("robot")} # 告诉函数去哪台机器人拿力矩数据
    )

@configclass
class TerminationsCfg:
    """终止条件配置"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    ee_too_far = DoneTerm(
        func=mdp.bad_ee_distance, 
        params={"threshold": 1, "command_name": "wiping", "robot_cfg": SceneEntityCfg("robot", body_names=".*_Link(11|22)")}
    )
    force_limit = DoneTerm(
        func=mdp.excessive_force, 
        params={"threshold": 60.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Link(11|22)")}
    )

##
# Environment configuration
##

@configclass
class WipingEnvCfg(ManagerBasedRLEnvCfg):
    """擦拭环境主配置"""
    scene: WipingSceneCfg = WipingSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # 1. 必须补全机器人资产 (解决 TypeError: Missing values detected in scene.robot)
        # 这里引用你从 robot_lab.assets.booster_gripper 导入的配置
        self.scene.robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2. 必须补全指令重采样时间 (解决 TypeError: Missing values detected in commands.wiping.resampling_time_range)
        # (1.0e9, 1.0e9) 代表在 20 亿秒内不重新生成轨迹，适合打磨这种固定轨迹任务
        self.commands.wiping.resampling_time_range = (1.0e9, 1.0e9)

        # 3. 其他仿真基础设置
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 0.005 # 200Hz 物理频率
        self.sim.use_fabric = False
        # 4. 视角设置
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.asset_name = "robot" # 告诉渲染器默认盯在哪儿