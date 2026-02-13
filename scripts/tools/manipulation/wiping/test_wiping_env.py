# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import argparse
from isaaclab.app import AppLauncher

# 1. 启动仿真 App
parser = argparse.ArgumentParser(description="Test Wiping - EE Pose Control Test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation, RigidObject
from robot_lab.tasks.manager_based.manipulation.wiping.wiping_env_cfg import WipingEnvCfg
import omni.usd 
from pxr import Usd, UsdGeom, Gf

# ==============================================================================
# ️ 视觉同步工具
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
    for asset in env.scene.articulations.values():
        base_path = f"/World/envs/env_{env_idx}/Robot"
        _apply_visual_transform(stage, base_path, torch.tensor([0.,0.,0.]), torch.tensor([1.,0.,0.,0.]), is_root=True)
        for i, body_name in enumerate(asset.data.body_names):
            _apply_visual_transform(stage, f"{base_path}/{body_name}", asset.data.body_pos_w[env_idx, i], asset.data.body_quat_w[env_idx, i])
    for asset in env.scene.rigid_objects.values():
        actual_path = f"/World/envs/env_{env_idx}/TargetBlock"
        _apply_visual_transform(stage, actual_path, asset.data.root_pos_w[env_idx], asset.data.root_quat_w[env_idx], is_root=True)

# ==============================================================================
# 易 傻瓜式 P 控制器
# ==============================================================================
def heuristic_policy(env, robot, target_pos, current_eef_idx):
    """
    Action Space: [dx, dy, dz, dRx, dRy, dRz, stiffness]
    """
    # 1. 获取当前末端位置
    current_eef_pos = robot.data.body_pos_w[0, current_eef_idx]
    
    # 2. 计算位置误差
    pos_error = target_pos - current_eef_pos
    
    # 3. 生成动作 (P控制)
    # 增大 Kp 以测试机器人是否有力气
    kp = 5.0 
    action_pos = torch.clamp(pos_error * kp, -0.1, 0.1) # 限制单步最大位移
    
    # 4. 组装完整动作
    total_dim = env.action_manager.total_action_dim
    actions = torch.zeros((env.num_envs, total_dim), device=env.device)
    # 位移
    actions[:, :3] = action_pos
    
    # ⚠️ 强制刚度满载：测试机器人是否还“不举”
    # 如果 action range 是 [0, 1]，设为 1.0 (最硬)
    # 如果 action range 是 [-1, 1]，也设为 1.0
    if total_dim >= 7:
        actions[:, 6] = 1.0 
    return actions, pos_error

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
    # ⚠️ 请确保这里是正确的末端 Link 名字
    target_link_name = "right_Link22"  
    try:
        eef_idx = robot.data.body_names.index(target_link_name)
    except ValueError:
        print(f"❌ 找不到 link: {target_link_name}，默认使用最后一个 link")
        eef_idx = len(robot.data.body_names) - 1

    contact_sensor = env.scene["contact_forces"]
    
    # 记录初始位置，作为参考点
    initial_eef_pos = robot.data.body_pos_w[0, eef_idx].clone()
    print(f" 初始末端位置: {initial_eef_pos.cpu().numpy()}")

    step_count = 0
    
    while simulation_app.is_running():
        # ---------------------------------------------------------
        # 1. 设定测试目标点 (Test Trajectory)
        # ---------------------------------------------------------
        target_pos = initial_eef_pos.clone()
        
        # 定义测试阶段
        phase = (step_count // 100) % 4 
        phase_name = "Hold"
        
        if phase == 0:
            # 保持原地 (看看会不会掉下来)
            phase_name = " 保持原地 (Hold)"
        elif phase == 1:
            # 往前伸 0.2米
            target_pos[0] += 0.2
            phase_name = "➡️ 向前伸 (X+0.2)"
        elif phase == 2:
            # 往左移 0.1米 (保持前伸)
            target_pos[0] += 0.2
            target_pos[1] += 0.1
            phase_name = "⬅️ 向左移 (Y+0.1)"
        elif phase == 3:
            # 往下压 0.1米
            target_pos[0] += 0.2
            target_pos[1] += 0.1
            target_pos[2] -= 0.1
            phase_name = "⬇️ 向下压 (Z-0.1)"

        # ---------------------------------------------------------
        # 2. 执行策略
        # ---------------------------------------------------------
        actions, error = heuristic_policy(env, robot, target_pos, eef_idx)
        env.step(actions)
        
        # ---------------------------------------------------------
        # 3. 视觉同步
        # ---------------------------------------------------------
        sync_scene_visuals(env, stage)
        
        # ---------------------------------------------------------
        # 4. 打印状态监控
        # ---------------------------------------------------------
        if step_count % 10 == 0:
            current_z = robot.data.body_pos_w[0, eef_idx, 2].item()
            target_z = target_pos[2].item()
            
            # 计算 L2 误差
            error_norm = torch.norm(error).item()
            
            # 状态指示符
            status_icon = "✅" if error_norm < 0.02 else "⚠️ 偏差大"
            if error_norm > 0.1: status_icon = "❌ 无法到达"
            
            print(f"Step {step_count:03d} | {phase_name}")
            print(f"   目标: {target_pos.cpu().numpy()}")
            print(f"   当前: {robot.data.body_pos_w[0, eef_idx].cpu().numpy()}")
            print(f"   误差: {error_norm:.4f} m {status_icon}")
            
            # 如果误差很大且 Z 轴一直掉，说明刚度不够
            if error[2] > 0.05: # 目标比当前高 5cm 以上，说明抬不起来
                 print("    提示: 机器人似乎抬不起来，请检查 nominal_kp 是否过小！")

        env.render()
        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()