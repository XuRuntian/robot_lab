# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import argparse
from isaaclab.app import AppLauncher

# 1. 启动仿真 App
parser = argparse.ArgumentParser(description="Test Wiping - Force Control Demo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation, RigidObject
from robot_lab.tasks.manager_based.manipulation.wiping.wiping_env_cfg import WipingEnvCfg
import omni.usd 
from pxr import Usd, UsdGeom, Gf

# ==============================================================================
# ️ 视觉同步工具 (保持之前修复好的版本)
# ==============================================================================
def _apply_visual_transform(stage, prim_path, pos, quat, is_root=False):
    if ".*" in prim_path or "{" in prim_path: return 
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid(): return
    xform = UsdGeom.Xformable(prim)
    if is_root: xform.ClearXformOpOrder()
    
    translate_op = next((op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if not translate_op: translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(pos[0].item(), pos[1].item(), pos[2].item()))
    
    orient_op = next((op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if not orient_op: orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    orient_op.Set(Gf.Quatd(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()))

def sync_scene_visuals(env, stage, env_idx=0):
    """只做同步，减少打印干扰"""
    # 同步机器人
    for asset in env.scene.articulations.values():
        base_path = f"/World/envs/env_{env_idx}/Robot"
        _apply_visual_transform(stage, base_path, torch.tensor([0.,0.,0.]), torch.tensor([1.,0.,0.,0.]), is_root=True)
        for i, body_name in enumerate(asset.data.body_names):
            _apply_visual_transform(stage, f"{base_path}/{body_name}", asset.data.body_pos_w[env_idx, i], asset.data.body_quat_w[env_idx, i])
    # 同步方块
    for asset in env.scene.rigid_objects.values():
        actual_path = f"/World/envs/env_{env_idx}/TargetBlock"
        _apply_visual_transform(stage, actual_path, asset.data.root_pos_w[env_idx], asset.data.root_quat_w[env_idx], is_root=True)

# ==============================================================================
# 易 傻瓜式 P 控制器 (替代 IK)
# ==============================================================================
def heuristic_policy(env, robot, target_pos, current_eef_idx):
    """
    简单的追踪策略：计算 (目标 - 当前) 的差值，作为动作发送。
    Action Space: [dx, dy, dz, dRx, dRy, dRz, stiffness]
    """
    # 1. 获取当前末端位置
    current_eef_pos = robot.data.body_pos_w[0, current_eef_idx]
    
    # 2. 计算位置误差
    pos_error = target_pos - current_eef_pos
    
    # 3. 生成动作 (P控制)
    kp = 2.0 # 比例增益，越大跑得越快
    action_pos = torch.clamp(pos_error * kp, -0.1, 0.1) # 限制最大速度
    
    # 4. 组装完整动作
    # 假设动作维度是 7 (3位移 + 3旋转 + 1刚度)
    total_dim = env.action_manager.total_action_dim
    actions = torch.zeros((env.num_envs, total_dim), device=env.device)
    
    # 填入位移指令 (前3维)
    actions[:, :3] = action_pos
    
    # 填入刚度指令 (假设第7维是刚度，范围通常是 [0, 1] 或 [-1, 1])
    # 设为 1.0 (硬) 以确保有力气下压
    if total_dim >= 7:
        actions[:, 6] = 1.0 
        
    return actions

# ==============================================================================
#  主程序
# ==============================================================================
def main():
    env_cfg = WipingEnvCfg()
    env_cfg.scene.num_envs = 1 
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    stage = omni.usd.get_context().get_stage()
    
    print("[INFO] 环境启动成功...")
    env.reset()
    
    # -------------------------------------------------------------
    # ️‍♂️ 自动寻找末端执行器索引
    # -------------------------------------------------------------
    robot = env.scene["robot"]
    # 这里请根据打印结果修改名字！比如 "link_6", "right_hand", "flange" 等
    # 常见的末端名字： "link_6", "right_Link6", "panda_hand", "tool0"
    print(f"机器人所有 Link: {robot.data.body_names}")
    
    # ⚠️【请修改这里】⚠️ 找到你的末端 Link 名字
    target_link_name = "right_Link22"  # 假设是这个，不对请改！
    try:
        eef_idx = robot.data.body_names.index(target_link_name)
    except ValueError:
        print(f"❌ 找不到 link: {target_link_name}，默认使用最后一个 link")
        eef_idx = len(robot.data.body_names) - 1

    apple = env.scene["apple"]
    contact_sensor = env.scene["contact_forces"]
    
    step_count = 0
    
    while simulation_app.is_running():
        # ---------------------------------------------------------
        # 1. 定义目标位置 (Target Strategy)
        # ---------------------------------------------------------
        # 拿到方块的真实物理位置
        target_pos = apple.data.root_pos_w[0].clone()
        
        # 阶段 1 (前100步): 移动到方块正上方 15cm 处
        if step_count < 100:
            target_pos[2] += 0.15
            
        # 阶段 2 (100步后): 垂直下压！目标设为方块表面以下 2cm
        else:
            # 假设方块高度 0.1m, 中心在 z，表面在 z + 0.05
            # 我们想压进去，所以目标设为 z + 0.03
            target_pos[2] += 0.03 

        # ---------------------------------------------------------
        # 2. 计算并执行动作
        # ---------------------------------------------------------
        actions = heuristic_policy(env, robot, target_pos, eef_idx)
        env.step(actions)
        
        # ---------------------------------------------------------
        # 3. 视觉同步
        # ---------------------------------------------------------
        sync_scene_visuals(env, stage)
        
        # ---------------------------------------------------------
        # 4. 力控数据监控 (完美适配正则匹配)
        # ---------------------------------------------------------
        if step_count % 10 == 0:
            # 1. 你想看哪个指尖的力？(通常是右指尖或者左指尖)
            # 必须是你上面正则能匹配到的名字！a
            target_tip_name = "right_Link22" 
            
            # 2. 安全查找：去传感器列表里找这个名字排第几
            if target_tip_name in contact_sensor.body_names:
                print(f"contact_sensor 监控了 {contact_sensor.body_names}")
                # 找到它在传感器数据中的索引 (比如可能是 0, 1, 2, 3 中的一个)
                sensor_idx = contact_sensor.body_names.index(target_tip_name)
                
                # 3. 取值
                net_forces = contact_sensor.data.net_forces_w[0, sensor_idx]
                force_z = abs(net_forces[2].item())
                
                # 4. 打印
                current_z = robot.data.body_pos_w[0, eef_idx, 2].item()
                target_z = target_pos[2].item()
                
                print(f"Step {step_count:03d} | 高度: {current_z:.3f} | 力(Link22): {force_z:.2f} N")
                
                if force_z > 0.5: 
                    print(" 接触到了！")
            else:
                # 调试专用：如果名字写错了，打印出来看看传感器到底监控了谁
                print(f"⚠️ 没找到 {target_tip_name}！传感器监控列表: {contact_sensor.data.body_names}")

        env.render()
        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()