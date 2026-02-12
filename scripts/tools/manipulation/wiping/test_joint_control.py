# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import argparse
from isaaclab.app import AppLauncher

# 1. 启动仿真 App
parser = argparse.ArgumentParser(description="Test All Joints Physics")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from robot_lab.tasks.manager_based.manipulation.wiping.wiping_env_cfg import WipingEnvCfg

def main():
    # 2. 配置环境
    env_cfg = WipingEnvCfg()
    env_cfg.scene.num_envs = 1  # 测试只需 1 个环境
    
    print("[INFO] 正在启动全关节测试环境...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    
    robot = env.scene["robot"]
    num_joints = robot.num_joints
    joint_names = robot.data.joint_names
    print(f"[INFO] 机器人加载成功，总关节数: {num_joints}")

    # 3. 设定目标：让所有关节在默认姿态基础上偏移 0.3 弧度
    # 提示：有些关节如果是限位很窄，0.3 可能会触发约束，可以根据需要调小
    test_offset = 0.3 
    default_pos = robot.data.default_joint_pos.clone()
    target_pos = default_pos + test_offset

    print(f" 全关节测试开始：所有关节目标偏移 +{test_offset} rad")
    
    dt = env.cfg.sim.dt
    step_count = 0

    while simulation_app.is_running():
        # A. 下发全身指令
        robot.set_joint_position_target(target_pos)
        robot.set_joint_effort_target(torch.zeros_like(target_pos))

        # B. 物理仿真步进
        robot.write_data_to_sim()
        env.sim.step()
        
        # C. 更新 Buffer (Isaac Lab 必需)
        robot.update(dt)
        
        # ------------------------------------------------------------------
        #  监控逻辑：每 50 步汇总一次“拖后腿”的关节
        # ------------------------------------------------------------------
        if step_count % 50 == 0:
            current_pos = robot.data.joint_pos[0] # 取第 0 个环境
            errors = torch.abs(current_pos - target_pos[0])
            
            # 找到误差最大的前 5 个关节
            max_errors, top_indices = torch.topk(errors, k=min(5, num_joints))
            
            print(f"\n--- Step {step_count:04d} 状态快照 ---")
            mean_error = torch.mean(errors).item()
            print(f" 全身平均误差: {mean_error:.4f}")
            
            print("❌ 误差最大的关节 Top 5 (检查其 Stiffness/Effort):")
            for i in range(len(top_indices)):
                idx = top_indices[i].item()
                name = joint_names[idx]
                err_val = max_errors[i].item()
                curr_val = current_pos[idx].item()
                
                status = " 瘫痪" if err_val > 0.2 else " 虚弱" if err_val > 0.05 else " 正常"
                print(f"   [{idx:02d}] {name:30} | 误差: {err_val:.3f} | 状态: {status}")

            if mean_error < 0.02:
                print("✨ 结论: 全身控制良好，动力学配置足以克服重力。")
            elif mean_error > 0.3:
                 print(" 警告: 机器人可能已塌陷，请检查 booster_gripper.py 中的 ImplicitActuatorCfg！")

        step_count += 1

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()