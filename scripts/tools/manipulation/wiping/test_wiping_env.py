# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import argparse
# 1. 必须首先导入 AppLauncher
from isaaclab.app import AppLauncher

# 2. 设置命令行参数并启动仿真 App
parser = argparse.ArgumentParser(description="Test Wiping Environment")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli) # 只有这行执行完，pxr 等库才可用
simulation_app = app_launcher.app

# 3. 在 App 启动后，再导入其他的库
import torch
from isaaclab.envs import ManagerBasedRLEnv
# 注意：确保你的 robot_lab 已经安装在 python 环境中，或者路径正确
from robot_lab.tasks.manager_based.manipulation.wiping.wiping_env_cfg import WipingEnvCfg

def main():
    # 1. 加载配置
    env_cfg = WipingEnvCfg()
    env_cfg.scene.num_envs = 1 # 测试阶段只开一个环境，方便观察
    
    # 2. 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("[INFO] 场景加载成功。开始交互测试...")
    
    # 3. 循环仿真
    obs, _ = env.reset()
    while simulation_app.is_running():
        # 生成随机动作 (7维: 6D位姿增量 + 1D刚度缩放)
        # 这里验证的是 actions.py 对 MotorCmd 的映射逻辑
        action_dim = sum(env.action_manager.action_term_dim)
        actions = torch.randn((env.num_envs, action_dim), device=env.device)
        
        # 执行动作并获取反馈 (验证 observation.py 对 LowState 的对齐)
        obs, rew, terminated, truncated, info = env.step(actions)
        
        # 渲染
        env.render()

if __name__ == "__main__":
    main()
    # 4. 测试结束关闭 App
    simulation_app.close()