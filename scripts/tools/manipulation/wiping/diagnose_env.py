# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import argparse
from isaaclab.app import AppLauncher

# 1. 启动仿真 App
parser = argparse.ArgumentParser(description="Robot Actuator Diagnostics")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from robot_lab.tasks.manager_based.manipulation.wiping.wiping_env_cfg import WipingEnvCfg

def diagnose_robot(env):
    robot = env.scene["robot"]
    
    print("\n" + "="*80)
    print(" 机器人致动系统深度诊断报告")
    print("="*80)
    
    # ------------------------------------------------------------------
    # 2. 刚度与阻尼检查 (核心修正点)
    # ------------------------------------------------------------------
    print("\n[2] 关节物理属性检查 (Stiffness & Damping)")
    print("-" * 80)
    
    # 修正属性名称：使用 default_joint_stiffness / default_joint_damping
    if hasattr(robot.data, "default_joint_stiffness"):
        stiffness = robot.data.default_joint_stiffness[0]
        damping = robot.data.default_joint_damping[0]
    else:
        # 兼容旧版本 (虽然报错提示不是这个，但以防万一)
        stiffness = robot.data.default_stiffness[0]
        damping = robot.data.default_damping[0]
    
    print(f"{'ID':<4} | {'Joint Name':<30} | {'Stiffness':<10} | {'Damping':<10} | {'Status'}")
    print("-" * 80)
    
    zero_stiffness_count = 0
    
    for i, name in enumerate(robot.data.joint_names):
        k = stiffness[i].item()
        d = damping[i].item()
        
        # 判断状态
        status = "✅ OK"
        if k < 1e-4:
            status = "⚠️ ZERO (软)"
            zero_stiffness_count += 1
            # 重点检查手臂关节 (Shoulder, Elbow, Wrist)
            if any(key in name for key in ["Shoulder", "Elbow", "Wrist", "Hand"]):
                status = "❌ DANGER (手臂无力!)"
        
        print(f"{i:<4} | {name:<30} | {k:<10.1f} | {d:<10.1f} | {status}")

    print("-" * 80)

    # ------------------------------------------------------------------
    # 3. 诊断结论
    # ------------------------------------------------------------------
    print("\n 诊断结论:")
    if zero_stiffness_count > 0:
        print(f"⚠️ 发现 {zero_stiffness_count} 个关节的基础刚度为 0.0！")
        print("  -> 这意味着对应的关节完全依赖 'ActionsCfg' 里的 P 控制器。")
        print("  -> 如果你在 ActionsCfg 里设的 nominal_kp 太小 (如 40)，机器人就会不举。")
    else:
        print("✅ 所有关节都有基础刚度。")

    print("="*80 + "\n")

def main():
    env_cfg = WipingEnvCfg()
    env_cfg.scene.num_envs = 1 
    
    # ---------------------------------------------------------
    #  想要测试修复效果？取消下面几行的注释！
    # ---------------------------------------------------------
    # print("[PATCH] 尝试动态修复手臂刚度...")
    # if "arms" in env_cfg.scene.robot.actuators:
    #     env_cfg.scene.robot.actuators["arms"].stiffness = 200.0
    #     env_cfg.scene.robot.actuators["arms"].damping = 10.0
    
    print("[INFO] 正在加载环境...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    
    diagnose_robot(env)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()