# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from robot_lab.assets.booster_gripper import BOOSTER_T1_CFG
# 引入你之前写好的通用配置
from robot_lab.tasks.manager_based.manipulation.wiping.wiping_env_cfg import WipingEnvCfg

@configclass
class BoosterT1WipingEnvCfg(WipingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 1. 场景配置覆盖：指定使用 Booster T1 机器人
        self.scene.robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 2. 动作限制覆盖：根据真机 MotorCmd 性能微调
        self.actions.arm_action.pose_limit = 0.01  # 擦拭动作不需要太快，减小步长
        self.actions.arm_action.nominal_kp = 40.0   # 对应电机默认刚度
        
        # 3. 任务指令覆盖：定义圆圈擦拭的中心点
        self.commands.wiping_cmd.center_range = {
            "x": (0.45, 0.55), 
            "y": (-0.1, 0.1)
        }
        self.commands.wiping_cmd.table_height = 0.73 # 匹配你的桌子高度

        # 4. 终止条件覆盖
        self.terminations.ee_too_far.threshold = 0.3 # 缩小容忍度，让机器人更精准
        
        # 5. 奖励权重微调
        self.rewards.force_tracking.weight = 5.0      # 加大力控的权重，这是课题核心
        self.rewards.pos_tracking.weight = 1.0