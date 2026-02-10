# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import argparse
from isaaclab.app import AppLauncher

# 1. 启动仿真 App
parser = argparse.ArgumentParser(description="Test Wiping Environment - Full Sync Fixed")
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
# ️ 核心工具：视觉变换写入
# ==============================================================================
def _apply_visual_transform(stage, prim_path, pos, quat, is_root=False):
    """安全写入变换，防止坐标偏移叠加"""
    if ".*" in prim_path or "{" in prim_path: return 

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid(): return
    
    xform = UsdGeom.Xformable(prim)
    if is_root:
        xform.ClearXformOpOrder() # 彻底清理旧的操作符，防止位移翻倍

    # 写入平移
    translate_op = next((op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if not translate_op: translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(pos[0].item(), pos[1].item(), pos[2].item()))
    
    # 写入旋转
    orient_op = next((op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if not orient_op: orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    orient_op.Set(Gf.Quatd(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()))

# ==============================================================================
#  全场景诊断与同步 (已修复 NameError)
# ==============================================================================
def sync_and_debug_scene(env, stage, step, env_idx=0):
    """同步并诊断场景。强制手动拉齐物理与视觉位姿。"""
    if step % 20 == 0:
        print(f"\n" + "="*50 + f"\n[Step {step}] 物理 vs 视觉 对齐报告\n" + "-"*50)

    # 1. 同步机器人 (Articulation)
    for asset in env.scene.articulations.values():
        base_path = f"/World/envs/env_{env_idx}/Robot"
        # 强制根节点视觉归零
        _apply_visual_transform(stage, base_path, torch.tensor([0.,0.,0.]), torch.tensor([1.,0.,0.,0.]), is_root=True)
        for i, body_name in enumerate(asset.data.body_names):
            _apply_visual_transform(stage, f"{base_path}/{body_name}", asset.data.body_pos_w[env_idx, i], asset.data.body_quat_w[env_idx, i])

    # 2. 同步刚体 (RigidObject - 你的 TargetBlock)
    for name, asset in env.scene.rigid_objects.items():
        # 手动拼接合法路径，避开通配符报错
        actual_path = f"/World/envs/env_{env_idx}/TargetBlock"
        
        phys_pos = asset.data.root_pos_w[env_idx].cpu().numpy()
        _apply_visual_transform(stage, actual_path, asset.data.root_pos_w[env_idx], asset.data.root_quat_w[env_idx], is_root=True)
        
        if step % 20 == 0:
            prim = stage.GetPrimAtPath(actual_path)
            if prim.IsValid():
                # 获取 USD 渲染器中的全局坐标
                v_pos = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
                # 修正：v_p 改为 v_pos
                error = torch.norm(torch.tensor(phys_pos) - torch.tensor(list(v_pos))).item()
                print(f"资产: {name} | 物理: {phys_pos[0]:.2f}, {phys_pos[1]:.2f}, {phys_pos[2]:.2f}")
                print(f"误差 ($L_2$): {error:.6f}")
    
    if step % 20 == 0: print("="*50)

# ==============================================================================
#  主程序
# ==============================================================================
def main():
    env_cfg = WipingEnvCfg()
    env_cfg.scene.num_envs = 1 
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    stage = omni.usd.get_context().get_stage()
    
    print("[INFO] 场景重置...")
    env.reset()
    step_count = 0
    
    while simulation_app.is_running():
        # 1. 物理步进
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        env.step(actions)
        
        # 2. 核心：同步并诊断
        sync_and_debug_scene(env, stage, step_count, env_idx=0)
        
        # 3. 渲染
        env.render()
        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()