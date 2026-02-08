# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import argparse
from isaaclab.app import AppLauncher

# 1. 启动仿真 App
parser = argparse.ArgumentParser(description="Test Wiping Environment - Precision Fixed")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import carb
from isaaclab.envs import ManagerBasedRLEnv
from robot_lab.tasks.manager_based.manipulation.wiping.wiping_env_cfg import WipingEnvCfg
import omni.usd 
from pxr import Usd, UsdGeom, UsdPhysics, Gf

# ==============================================================================
# 🖐️ 手动强制同步逻辑 (修正精度版)
# ==============================================================================
def manual_sync_robot_visuals(robot, stage, robot_root_path, env_idx=0):
    all_pos = robot.data.body_pos_w[env_idx]
    all_quat = robot.data.body_quat_w[env_idx]
    body_names = robot.data.body_names

    for i, body_name in enumerate(body_names):
        prim_path = f"{robot_root_path}/{body_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid(): continue
            
        xform = UsdGeom.Xformable(prim)
        
        # --- 安全写入平移 ---
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if not translate_op:
            translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        
        pos = all_pos[i]
        translate_op.Set(Gf.Vec3d(pos[0].item(), pos[1].item(), pos[2].item()))
        
        # --- 安全写入旋转 (修复 PrecisionFloat 错误) ---
        orient_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op
                break
        if not orient_op:
            # 显式指定双精度，匹配 Gf.Quatd
            orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        
        quat = all_quat[i]
        orient_op.Set(Gf.Quatd(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()))

# ==============================================================================
# 🧹 清理幽灵与错位检查 (保持不变)
# ==============================================================================
def clean_ghost_meshes(robot_prim_path_regex):
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_prim_path_regex)
    if not robot_prim.IsValid(): return
    for prim in Usd.PrimRange(robot_prim):
        if prim.IsA(UsdGeom.Mesh):
            is_physics_driven = False
            curr_parent = prim.GetParent()
            while curr_parent and curr_parent.GetPath() != robot_prim.GetPath():
                if curr_parent.HasAPI(UsdPhysics.RigidBodyAPI):
                    is_physics_driven = True
                    break
                curr_parent = curr_parent.GetParent()
            if not is_physics_driven and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdGeom.Imageable(prim).MakeInvisible()

def find_all_faulty_links(robot, robot_root_path, threshold=0.05):
    stage = omni.usd.get_context().get_stage()
    num_bodies = robot.data.body_pos_w.shape[1]
    faulty_links = []
    for i in range(num_bodies):
        body_name = robot.data.body_names[i]
        physx_pos = robot.data.body_pos_w[0, i].cpu().numpy()
        prim = stage.GetPrimAtPath(f"{robot_root_path}/{body_name}")
        if prim.IsValid():
            v_pos = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
            error = torch.norm(torch.tensor(physx_pos) - torch.tensor(list(v_pos))).item()
            if error > threshold: faulty_links.append((body_name, error))
    if not faulty_links: print("✅ 所有 Link 已同步")
    else: print(f"❌ 发现 {len(faulty_links)} 个错位 Link")

# ==============================================================================
# 🚀 主程序
# ==============================================================================
def main():
    env_cfg = WipingEnvCfg()
    env_cfg.scene.num_envs = 1 
    
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    stage = omni.usd.get_context().get_stage()
    robot_prim_path = "/World/envs/env_0/Robot"
    
    print("[INFO] 正在执行初始化修复...")
    
    # 关键修复点：重置根节点变换 (处理精度冲突)
    robot_root_prim = stage.GetPrimAtPath(robot_prim_path)
    if robot_root_prim.IsValid():
        xform = UsdGeom.Xformable(robot_root_prim)
        xform.ClearXformOpOrder()
        # 统一使用 PrecisionDouble 避免报错
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    
    # 清理视觉锁
    for prim in Usd.PrimRange(robot_root_prim):
        if prim.IsA(UsdGeom.Xformable):
            UsdGeom.Xformable(prim).SetResetXformStack(False)
    
    clean_ghost_meshes(robot_prim_path)
    env.reset()
    
    step_count = 0
    while simulation_app.is_running():
        actions = torch.zeros((env.num_envs, sum(env.action_manager.action_term_dim)), device=env.device)
        env.step(actions)
        
        # 核心：每帧同步
        manual_sync_robot_visuals(robot, stage, robot_prim_path, env_idx=0)
        
        if step_count % 60 == 0:
            find_all_faulty_links(robot, robot_root_path=robot_prim_path)
            
        env.render()
        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()