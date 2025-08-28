from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils.math import quat_error_magnitude, quat_mul

def object_is_lifted(object: RigidObject, ee_frame: FrameTransformer, std: float, std_height: float, desired_height: float) -> torch.Tensor:
    object_height_from_desired = desired_height - object.data.root_pos_w[:, 2]
    object_height_reward = 1 - torch.tanh(object_height_from_desired / std_height)

    reach_reward = object_position_error_tanh(object, ee_frame, std)
    reward =  object_height_reward * reach_reward

    #print(f"Reach reward: {reach_reward}, Object height: {object_height_from_desired}, Reward: {reward}")

    return reward

def action_rate_reward(previous_actions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:

    return torch.sum(torch.square(actions[:, -1].unsqueeze(-1) - previous_actions[:, -1].unsqueeze(-1)), dim=1)

def joint_vel_reward(ur5e_joint_vel: torch.Tensor, arm_joints_ids: tuple, gripper_joints_ids) -> torch.Tensor:

    return torch.sum(torch.square(ur5e_joint_vel[:, arm_joints_ids]), dim=1)

def gripper_reward(actions: torch.Tensor, object: RigidObject, ee_frame: FrameTransformer) -> torch.Tensor:

    distance_to_object = object_position_error(object, ee_frame)
    object_is_close = torch.where(distance_to_object < 0.03, 1.0, -1.0)

    gripper_action = actions[:, -1].unsqueeze(-1)
    gripper_closed = torch.where(gripper_action < 0, 1.0, -1.0).squeeze()
    
    reward = gripper_closed * object_is_close

    print(f"Gripper closed: {gripper_closed}, Distance: {distance_to_object}, Reward: {reward}")

    return reward

def object_moved_xy(object: RigidObject, original_object_pos: torch.Tensor) -> torch.Tensor:
    object_pos = object.data.root_pos_w
    original_pos = original_object_pos

    distance_moved = torch.norm(object_pos[:, :2] - original_pos[:, :2], dim=1)

    return distance_moved

def tray_moved(tray: RigidObject) -> torch.Tensor: 
    tray_vel = tray.data.root_vel_w
    tray_speed = torch.norm(tray_vel, dim=1)

    return tray_speed

def joint_2_tuning(ur5e_joint_pos: torch.Tensor, std: float) -> torch.Tensor:
    joint_2_pos = ur5e_joint_pos[:, 1]  # Joint 2 position
    reward = torch.tanh(-joint_2_pos / std) 
    
    return reward

def end_effector_orientation_error(ee_frame: FrameTransformer, std: float) -> torch.Tensor:
    number_of_envs = ee_frame.data.target_quat_w.shape[0]

    des_quat_w = torch.tensor([0.0, 0.0, 1.0, 0.0], device=ee_frame.device).repeat(number_of_envs, 1)
    curr_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    reward = torch.cos(error / std)

    print(f"Orientation error: {error}, Reward: {reward}")

    return reward

def object_position_error(object: RigidObject, ee_frame: FrameTransformer) -> torch.Tensor:
    cube_pos_w = object.data.root_pos_w  # Object position in world frame
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # End-effector position in world frame

    object_ee_distance = torch.norm(ee_pos_w - cube_pos_w, dim=1)
    return object_ee_distance

def object_position_error_tanh(object: RigidObject, ee_frame: FrameTransformer, std: float) -> torch.Tensor:
    object_ee_distance = object_position_error(object, ee_frame)
    reward = 1 - torch.tanh(object_ee_distance / std)
    return reward

def ground_hit_avoidance(object: RigidObject, ee_frame: FrameTransformer) -> torch.Tensor:
    cube_z_pos_w = object.data.root_pos_w[:, 2]
    ee_z_pos_w = ee_frame.data.target_pos_w[..., 0, 2]

    height = ee_z_pos_w - cube_z_pos_w

    reward = torch.where(height > 0.0, 1.0, 0.0)
    return reward