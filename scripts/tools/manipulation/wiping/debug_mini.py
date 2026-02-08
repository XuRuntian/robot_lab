# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import argparse
import carb
from isaaclab.app import AppLauncher

# 1. 启动 App
parser = argparse.ArgumentParser(description="Debug Limp Robot - Success")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import omni.usd

from robot_lab.assets.booster_gripper import BOOSTER_T1_CFG

# ==============================================================================
# 🖐️ 手动挡：强行同步 (修正坐标系版)
# ==============================================================================
def manual_sync_robot_visuals(robot, stage, sim_device):
    # 获取所有 Body 的世界坐标
    all_pos = robot.data.body_pos_w[0] 
    all_quat = robot.data.body_quat_w[0]
    body_names = robot.body_names
    
    for i, body_name in enumerate(body_names):
        prim_path = f"/World/Robot/{body_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
            
        pos = all_pos[i]
        quat = all_quat[i]
        
        xform = UsdGeom.Xformable(prim)
        
        # 写入平移
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if not translate_op:
            translate_op = xform.AddTranslateOp()
        
        # [关键修正] 直接写入世界坐标
        # 因为我们在 main() 里把父级 /World/Robot 设为了 (0,0,0)
        # 所以 Local == World，这样写就是对的！
        translate_op.Set(Gf.Vec3d(pos[0].item(), pos[1].item(), pos[2].item()))
        
        # 写入旋转
        orient_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op
                break
        if not orient_op:
            orient_op = xform.AddOrientOp()
        orient_op.Set(Gf.Quatd(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()))

# ==============================================================================
# 🚀 主程序
# ==============================================================================
def main():
    # 配置
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, use_fabric=False)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 1.0], [0.0, 0.0, 0.5])

    # 地面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground)
    
    # 3. 机器人配置 [关键修改]
    print("[INFO] 正在给机器人打麻药 (Stiffness=0)...")
    
    # 🔥🔥🔥 修正点：把根节点设为 0，防止双重叠加 🔥🔥🔥
    BOOSTER_T1_CFG.init_state.pos = (0.0, 0.0, 0.0)  # 改成 0
    # 我们改用手动抬高第一个关节的初始位置来制造下坠效果
    # 或者直接让它从 0 开始躺平，这不影响观察同步性
    
    for act_name, act_cfg in BOOSTER_T1_CFG.actuators.items():
        act_cfg.stiffness = 0.0
        act_cfg.damping = 0.0 

    # 生成机器人
    robot = Articulation(BOOSTER_T1_CFG.replace(prim_path="/World/Robot"))
    
    sim.reset()
    stage = omni.usd.get_context().get_stage()

    # 清理视觉锁
    print("🧹 清理视觉锁...")
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot")):
        if prim.IsA(UsdGeom.Xformable):
            UsdGeom.Xformable(prim).SetResetXformStack(False)

    # 手动把机器人整体抬高一点点 (通过物理瞬移)，这样它才有得掉
    # 注意：这只是为了视觉效果，不改 USD 父级
    root_idx = robot.find_bodies("AL6")[0][0] # 假设 AL6 是基座附近
    robot.write_root_pose_to_sim(torch.tensor([[0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
    robot.reset() # 刷新状态

    print("[INFO] 仿真开始！见证奇迹的时刻...")
    
    step = 0
    while simulation_app.is_running():
        robot.set_joint_effort_target(torch.zeros(robot.num_joints, device=sim.device))
        robot.write_data_to_sim()
        sim.step()
        robot.update(0.005)
        
        # 手动同步
        manual_sync_robot_visuals(robot, stage, sim.device)
        
        if step % 60 == 0:
            target_body = "AL6"
            phys_idx = robot.find_bodies(target_body)[0][0]
            phys_pos = robot.data.body_pos_w[0, phys_idx]
            
            vis_prim = stage.GetPrimAtPath(f"/World/Robot/{target_body}")
            vis_mat = UsdGeom.Xformable(vis_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            vis_pos = Gf.Vec3f(vis_mat.ExtractTranslation())
            
            # 计算误差
            err = torch.norm(phys_pos - torch.tensor(list(vis_pos), device=sim.device)).item()
            status = "✅ 完美同步!" if err < 0.05 else "❌ 仍有误差"
            
            print(f"[Step {step:04d}] 物理Z: {phys_pos[2]:.3f} | 视觉Z: {vis_pos[2]:.3f} | {status}")
            
        step += 1

if __name__ == "__main__":
    main()
    simulation_app.close()