
from scipy.spatial.transform import Rotation as R
import numpy as np
from utils.flow_pack.visualizer import visualize

def u_rot(roll, pitch, yaw, intrinsics, grid_size):
# def u_rot(yaw, pitch, roll, intrinsics, grid_size):

    a = pitch
    b = yaw
    g = roll
    f = intrinsics[0,0]
    H,W = grid_size
    cx, cy = intrinsics[0,2], intrinsics[1,2]

    (x, y) = np.meshgrid(np.arange(0, W, dtype=np.float32) - cx + 0.5,
                         np.arange(0, H, dtype=np.float32) - cy + 0.5)

    # Equations 1 and 2 from page 3 of Passive Navigation as a Pattern Recognition Problem
    u_rot = a*x*y/f - b*(x*x/f + f) + g*y
    v_rot = a*(y*y/f+f) - b*x*y/f - g*x

    return np.dstack((u_rot, v_rot))

def u_trans(v_x, v_y, v_z, depth, intrinsics):
    u = v_x
    v = v_y
    w = v_z
    Z = depth
    f = intrinsics[0,0]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    H,W = depth.shape
        
    (x, y) = np.meshgrid(np.arange(0, W, dtype=np.float32) - cx + 0.5,
                         np.arange(0, H, dtype=np.float32) - cy + 0.5)

    # Equations 1 and 2 from page 3 of Passive Navigation as a Pattern Recognition Problem
    # Substitute x_0 and y_0
    u_t = (w * x - u * f) / Z
    v_t = (w * y - v * f) / Z

    return np.dstack((u_t, v_t))

def compute_optical_flow(rpose, intrinsics, depth):
    flow_rot = u_rot(rpose[3], rpose[4], rpose[5], intrinsics, grid_size = depth.shape)
    flow_t = u_trans(rpose[0], rpose[1], rpose[2], depth,intrinsics)
    return flow_t + flow_rot

def visualize_optical_flow(flow):
    return visualize(flow)
# def visualize_optical_flow(flow):
#     theta = np.mod(np.arctan2(flow[:, :, 0], flow[:, :, 1]) + 2*np.pi, 2*np.pi)

#     flow_norms = np.linalg.norm(flow, axis=2)
#     flow_norms_normalized = flow_norms / np.max(flow_norms)

#     flow_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
#     flow_hsv[:, :, 0] = 180 * theta / (2*np.pi)
#     flow_hsv[:, :, 1] = 255 * flow_norms_normalized
#     flow_hsv[:, :, 2] = 255 * (flow_norms_normalized > 0)

#     return flow_hsv



def cam2ned(axis):
    return np.array((axis[2], axis[0], axis[1]))




def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)

    return intrinsicLayer