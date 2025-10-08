
import time
import torch

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from termcolor import colored
from pathlib import Path

import cv2
import os

os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['MUJOCO_GL'] = 'glfw'

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                        np.array([point1[0], point1[1], point1[2]]),
                        np.array([point2[0], point2[1], point2[2]]))

def key_call_back( keycode):
    global video_writer, recording, curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    if chr(keycode) == "R":
        print(colored("Reset", "red"))
        time_step = 0
    elif chr(keycode) == " ":
        print(colored("Paused", "green"))
        paused = not paused
    elif chr(keycode) == "N":
        print(colored("Next", "green"))
        if motion_id >= len(motion_data_keys) - 1:
            print(colored("End of Motion", "red"))
            motion_id = 0
        else:
            motion_id += 1
            curr_motion_key = motion_data_keys[motion_id]
            print(curr_motion_key)
    elif chr(keycode) == "V":
        # 切换录制状态
        recording = not recording
        if recording:
            print(colored("Started Recording", "yellow"))
        else:
            print(colored("Stopped Recording", "yellow"))
            if video_writer is not None:
                video_writer.release()
                video_writer = None
    else:
        print(colored(f"Not mapped: {chr(keycode)}", "red"))


@hydra.main(version_base=None, config_path="../../humanoidverse/config", config_name="base")
def main(cfg : DictConfig) -> None:
    global video_writer, recording, curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    device = torch.device("cpu")

    asset_path = Path(cfg.robot.motion.asset.assetRoot)
    asset_file = cfg.robot.motion.asset.assetFileName
    humanoid_xml = asset_path / asset_file

    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False

    recording = False
    video_writer = None
    # 视频设置
    video_width = 640
    video_height = 480
    fps = int(1/dt)  # 30 FPS
    output_dir = Path("videos")
    output_dir.mkdir(exist_ok=True)

    if cfg.visualize_motion_file is None:
        logger.error(colored("No motion file provided", "red"))
    else:
        visualize_motion_file = cfg.visualize_motion_file
    logger.info(colored(f"Visualizing Motion: {visualize_motion_file}", "green"))
    motion_data = joblib.load(visualize_motion_file)
    motion_data_keys = list(motion_data.keys())

    mj_model = mujoco.MjModel.from_xml_path(str(humanoid_xml))
    mj_data = mujoco.MjData(mj_model)

    mj_model.opt.timestep = dt

    # 创建离屏渲染器用于录制
    renderer = mujoco.Renderer(mj_model, height=video_height, width=video_width)
    # 设置录制摄像机参数
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    # 自定义摄像机视角 - 可根据需要调整
    camera.azimuth = 45        # 水平旋转角度 (0-360)
    camera.elevation = -20     # 垂直角度 (-90到90)
    camera.distance = 3.0      # 距离目标的距离
    camera.lookat = np.array([0.0, 0.0, 1.0])  # 观察目标点 [x, y, z]

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(24):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        print(f"Created {viewer.user_scn.ngeom} capsules")
        while viewer.is_running():
            step_start = time.time()
            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]
            curr_time = int(time_step/dt) % curr_motion['dof'].shape[0]
            
            mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]
            mj_data.qpos[7:] = curr_motion['dof'][curr_time]
                
            mujoco.mj_forward(mj_model, mj_data)

            # 录制视频
            if recording:
                if video_writer is None:
                    # 创建视频文件
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    video_path = output_dir / f"motion_{curr_motion_key}_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (video_width, video_height))
                    print(colored(f"Recording to: {video_path}", "yellow"))
                # 动态调整摄像机目标 - 跟踪机器人
                camera.lookat = mj_data.qpos[:3]  # 跟踪机器人根部位置
                # 渲染帧
                renderer.update_scene(mj_data, camera=camera)
                frame = renderer.render()
                # OpenCV使用BGR格式
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            if not paused:
                time_step += dt

            if 'smpl_joints' in motion_data[curr_motion_key].keys():
                joint_gt = motion_data[curr_motion_key]['smpl_joints']
                
                # for i in range(joint_gt.shape[1]):
                #     viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    # 清理资源
    if video_writer is not None:
        video_writer.release()
    renderer.close()


if __name__ == "__main__":
    main()