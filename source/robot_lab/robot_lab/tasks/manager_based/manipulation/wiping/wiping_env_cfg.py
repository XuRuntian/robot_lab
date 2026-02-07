from __future__ import annotations

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
from robot_lab.assets.booster_gripper import BOOSTER_T1_CFG

##
# Scene Definition
##

@configclass
class WipingSceneCfg(InteractiveSceneCfg):
    """配置擦拭任务的场景，包含机器人、桌子和目标表面"""
    
    # 1. 地面与灯光
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    # 2. 机器人 (使用你定义的 Booster T1)
    robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 3. 擦拭物体 (例如桌上的苹果或圆球)
    target_object = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.75)),
    )
    
    # 4. 接触传感器 (对齐真机 LowState 的力反馈需求)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_Hand_Roll", history_length=3, debug_vis=True
    )

##
# MDP Settings
##

@configclass
class CommandsCfg:
    """定义擦拭轨迹指令"""
    wiping_cmd = mdp.WipingCommandCfg() # 使用你写的圆形轨迹指令

@configclass
class ActionsCfg:
    """定义对齐 MotorCmd 的变阻抗动作项"""
    # 7维 Action: 6D Pose + 1D Kp Scale
    arm_action = mdp.WipingVariableImpedanceActionCfg(
        asset_name="robot",
        pose_limit=0.02,
        nominal_kp=40.0,
        damping_ratio=1.0
    )

@configclass
class ObservationsCfg:
    """对齐 LowState 文档的观测配置"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        # 关节状态 (q, dq)
        joint_pos = ObsTerm(func=mdp.joint_pos_raw, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_raw, noise=Unoise(n_min=-0.1, n_max=0.1))
        
        # 估算力矩 (tau_est)
        joint_tau = ObsTerm(func=mdp.joint_tau_est_raw)
        
        # 任务目标与偏差
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "wiping_cmd"})
        rel_ee_error = ObsTerm(func=mdp.ee_position_error_b, params={"command_name": "wiping_cmd", "robot_cfg": SceneEntityCfg("robot", body_names=".*_Hand_Roll")})
        
        # IMU 数据
        imu_rpy = ObsTerm(func=mdp.robot_imu_rpy)
        
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """环境随机化 (Sim2Real 的基础)"""
    # 模拟校准误差
    randomize_calibration = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_distribution_params": (-0.02, 0.02),
            "operation": "add",
        },
    )
    # 每次重置随机化物体位置
    reset_target = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("target")}
    )

@configclass
class RewardsCfg:
    """评价打磨质量的奖励函数"""
    # 1. 追踪奖励 (指数型)
    pos_tracking = RewTerm(func=mdp.ee_position_tracking_exp, weight=1.0, params={"std": 0.2, "command_name": "wiping_cmd", "robot_cfg": SceneEntityCfg("robot", body_names=".*_Hand_Roll")})
    
    # 2. 恒力奖励 (核心)
    force_tracking = RewTerm(func=mdp.constant_force_tracking_exp, weight=2.0, params={"std": 2.0, "target_force": 5.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Hand_Roll")})
    
    # 3. 惩罚项 (保护电机)
    torque_penalty = RewTerm(func=mdp.joint_torque_penalty, weight=-0.001)

@configclass
class TerminationsCfg:
    """终止条件：防撞、防飞、防过载"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    ee_too_far = DoneTerm(func=mdp.bad_ee_distance, params={"threshold": 0.5, "command_name": "wiping_cmd", "robot_cfg": SceneEntityCfg("robot", body_names=".*_Hand_Roll")})
    force_limit = DoneTerm(func=mdp.excessive_force, params={"threshold": 50.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Hand_Roll")})

##
# Environment Configuration
##

@configclass
class WipingEnvCfg(ManagerBasedRLEnvCfg):
    """完整的擦拭环境配置"""
    scene: WipingSceneCfg = WipingSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4 # 控制频率 50Hz (200Hz / 4)
        self.episode_length_s = 10.0
        self.sim.dt = 0.005 # 200Hz 物理步长
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (1.5, 0.0, 1.5)
        self.viewer.asset_name = "robot"