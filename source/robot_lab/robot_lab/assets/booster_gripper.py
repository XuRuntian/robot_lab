# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

# 确保路径引用正确
from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR



BOOSTER_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/booster/t1_description/usd/t1_with_7dof_arms_gripper.usd",
        # 激活接触传感器（这是解决你之前 ValueError 的关键）
        activate_contact_sensors=True,
        # 刚体属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        # 关节根属性：增加迭代次数以提高变阻抗控制的数值稳定性
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),

    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7), # 略高于地面以对齐桌子高度
        joint_pos={
            # Head
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            # Arm
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            # Waist
            "Waist": 0.0,
            # Leg
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.42,
            ".*_Ankle_Pitch": -0.23,
            ".*_Ankle_Roll": 0.0,
        },
    ),
    # 允许 90% 的物理限位，防止打磨时卡死
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_Hip_.*", ".*_Knee_.*", "Waist"],
            stiffness=200.0,
            damping=5.0,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_.*"],
            stiffness=50.0,
            damping=1.0,
        ),
        # 臂部：设为 0 是为了让 actions.py 中的变阻抗逻辑（Variable Impedance）完全接管
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_Shoulder_.*", ".*_Elbow_.*", ".*_Wrist_.*", ".*_Hand_.*",],
            stiffness=100.0, 
            damping=5.0,
        ),
        "head": ImplicitActuatorCfg( # 补全头部关节，凑足 37 个
            joint_names_expr=["Head_.*", "AAHead_.*"],
            stiffness=10.0,
            damping=1.0,
        ),
        # 夹爪：高刚度确保握死工具
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*_Link22", ".*_Link11"], 
            stiffness=1000.0,       
            damping=10.0,
        ),
        "extra_links": ImplicitActuatorCfg(
            # 补全剩下的 Link1, Link2 (手指根部) = 4个
            # 加上这组，总数就对齐 37 了
            joint_names_expr=["left_Link[12]$", "right_Link[12]$"], 
            stiffness=100.0,
            damping=5.0,
        )
    },
)