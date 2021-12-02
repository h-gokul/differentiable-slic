###############################################################################
#
# File: normal_flow.py
#
# Functions to calculate normal flow
#
# History:
# 02-16-20 - Levi Burner - Created file
#
###############################################################################

import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation
from utils.flow_pack.visualizer import visualize

# Do any elementry image processing to prepare the image
# Right now just Gaussian blur
def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def prepare_image(im, kernel_size=5):    
    blur = cv2.GaussianBlur(im, (kernel_size, kernel_size), 0)
    return blur

# Calculate the normal flow using Sobel kernel for spatial derivative
# and finite difference for temporal derivative. Threshold based on magnitude
# to create a mask representing which vectors have a signed projection that is
# likely to be correct
# TODO: configurable derivative filters
# TODO: configurable significance mask generation
# Returns gradient, optical flow projection onto gradient, and significance mask
def normal_flow_1_channel(im, im_prev):
    """
    Returns the gradient 

    Args:
        im ([type]): [description]
        im_prev ([type]): [description]

    Returns:
        [type]: [description]
    """
    # compute finite temporal difference
    dI = im.astype(np.float32) - im_prev.astype(np.float32)
    dI_scaled = dI * 1.0/255.0 # Scale from -255 to 255 to -1.0 to 1.0
    dI_mag = np.abs(dI_scaled) 

    # # compute spatial gradients
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT) 
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT) 
    # window = (np.array([1, -8, 0, 8, -1])/12).reshape(1, -1)
    # grad_x = cv2.filter2D(np.float32(im), -1,  window, cv2.BORDER_REFLECT)
    # grad_y = cv2.filter2D(np.float32(im), -1,  window.T, cv2.BORDER_REFLECT)
    
    grad = np.dstack((grad_x, grad_y))

    # compute magnitude of spatial gradient
    grad_scaled = grad * 1.0/255.0 # Scale from -255 to 255 to -1.0 to 1.0
    grad_mag = np.linalg.norm(grad_scaled, axis=2) # h,w

    # TODO no magic numbers like this, these are really noise parameters of the camera
    gradient_threshold = 1.0/255.0
    intensity_threshold = 1.0/255.0

    # filter out gradient/difference maps based on thresholds
    grad_significant = grad_mag > gradient_threshold
    dI_significant = np.logical_and(dI_mag > intensity_threshold, dI_mag < grad_mag / 2.0)
    flow_significant = dI_significant * grad_significant

    # (nx, ny) = unit spatial gradient vector =  gradients/magnitude
    grad_unit = np.divide(grad_scaled, np.atleast_3d(grad_mag), where=np.atleast_3d(flow_significant))
    # u_n = scalar normal flow = temporal difference/ gradient magnitude 
    flow_projection = np.divide(-dI_scaled, grad_mag, where=flow_significant)

    grad_unit[flow_significant==False, :] = 0
    flow_projection[flow_significant==False] = 0

    return (grad_unit, flow_projection)

def create_normal_flow_list(grad_unit, flow_projection):
    index_list = np.nonzero(flow_projection)
    grad_unit_list = array_to_list(index_list, grad_unit)
    flow_projection_list = array_to_list(index_list, flow_projection)

    return (index_list, grad_unit_list, flow_projection_list)

def array_to_list(index_list, array):
    field_list = array[index_list[0], index_list[1]]
    return field_list

def list_to_array(shape, index_list, signed_list):
    array = np.zeros(shape)
    array[index_list] = signed_list
    return array

# Generate an HSV image using color to represent the gradient direction in a normal flow field
def visualize_normal_flow(grad_unit, flow_projection):
    normal_flow = np.multiply(grad_unit, np.atleast_3d(flow_projection))
    theta = np.mod(np.arctan2(normal_flow[:, :, 0], normal_flow[:, :, 1]) + 2*np.pi, 2*np.pi)

    flow_hsv = np.zeros((flow_projection.shape[0], flow_projection.shape[1], 3), dtype=np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255 * (np.abs(flow_projection) > 0)#* np.abs(flow_projection)
    flow_hsv[:, :, 2] = 255 * (np.abs(flow_projection) > 0)

    return flow_hsv


def rescale_intrinsics(intrinsics, scalex, scaley):
    intrinsics[0] *=  scalex
    intrinsics[1] *=  scaley
    return intrinsics


def projected_normal_flow(im, flow):
    window = (np.array([1, -8, 0, 8, -1])/12).reshape(1, -1)
    grad_x = cv2.filter2D(np.float32(im), -1,  window, cv2.BORDER_REFLECT)
    grad_y = cv2.filter2D(np.float32(im), -1,  window.T, cv2.BORDER_REFLECT)

    grad_mag = np.sqrt(grad_x**2+grad_y**2)  # pred h,w
    grad_x[grad_mag<20] = 0
    grad_y[grad_mag<20] = 0

    NGTx = grad_x * (flow[:,:,0] * grad_x + flow[:,:,1] * grad_y)/(grad_x**2+grad_y**2)
    NGTy = grad_y * (flow[:,:,0] * grad_x + flow[:,:,1] * grad_y)/(grad_x**2+grad_y**2)
    NGTx[np.isnan(NGTx)]=0;  NGTy[np.isnan(NGTy)]=0
    nflow = np.dstack((NGTx, NGTy))

    return nflow


def visualize_projected_normal_flow(nflow):
    nflow_mag = np.sqrt(nflow[:,:,0]**2 + nflow[:,:,1]**2)
    theta = np.mod(np.arctan2(nflow[:, :, 0], nflow[:, :, 1]) + 2*np.pi, 2*np.pi)
    flow_hsv = np.zeros((nflow.shape[0], nflow.shape[1], 3), dtype=np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255 * (np.abs(nflow_mag) > 0)#* np.abs(flow_projection)
    flow_hsv[:, :, 2] = 255 * (np.abs(nflow_mag) > 0)
    return flow_hsv

# def normalflow_mag_dir(flow, im0,device):
#     # nflow = projected_normalflow(im0, flow, device)
#     nflow = nflow.permute(0,2,3,1)
#     u_n_scalar = torch.sqrt(nflow[...,:1]**2 + nflow[...,1:]**2)
#     grad_n = nflow/u_n_scalar
#     grad_n[torch.isnan(grad_n)]=0
#     return grad_n, u_n_scalar

# Generate an HSV image representing the color associated with the direction of a vector in flow field
def flow_direction_image(shape=(60,60)):
    (x, y) = np.meshgrid(np.arange(0, shape[1]) - shape[1]/2,
                         np.arange(0, shape[0]) - shape[0]/2)
    theta = np.mod(np.arctan2(x, y) + 2*np.pi, 2*np.pi)

    flow_hsv = np.zeros((shape[0], shape[1], 3)).astype(np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255
    flow_hsv[:, :, 2] = 255
    return flow_hsv

# Rescale flow function
def rescale_flow(flow, shape):
    out_h, out_w = shape
    x_scaling = out_w / flow.shape[1]
    y_scaling = out_h / flow.shape[0]
    # flow = flow.cpu().numpy().transpose(1, 2, 0)

    flow = cv2.resize(flow, (out_w, out_h), interpolation=cv2.INTER_AREA)

    flow[:, :, 0] = flow[:, :, 0] * x_scaling
    flow[:, :, 1] = flow[:, :, 1] * y_scaling
    return flow
    # return torch.tensor(flow.transpose(2, 0, 1))

def visualize_optical_flow(flow):
    return visualize(flow)

# Calculate an optical flow field due to a body rate rotation
# alpha is pitch;   # beta is yaw;  # gamma is roll

# Bmat[:, 0, 0] = (x*y)/fy;                 Bmat[:, 0, 1] = (-(x*x)/fx).sub(fx);   Bmat[:, 0, 2] = y*(fx/fy)
# Bmat[:, 1, 0] = ((y*y)/fy).add(fy);       Bmat[:, 1, 1] = -x*y/(fx);        Bmat[:, 1, 2] = -x*(fy/fx)

################################################
# def u_rot(yaw, pitch, roll, intrinsics, grid_size):
#     a = pitch
#     b = yaw
#     g = roll
#     # a = yaw
#     # b = pitch
#     # g = roll

#     fx, fy = intrinsics[0,0], intrinsics[1,1]
#     cx, cy = intrinsics[0,2], intrinsics[1,2]
#     (x, y) = np.meshgrid(np.arange(0, grid_size[1], dtype=np.float32) - (cx + 0.5),
#                          np.arange(0, grid_size[0], dtype=np.float32) - (cy + 0.5)
#                          )
#     u_rot = a*(x*y/fy) - b*(x*x/fx+ fx) + g*y*(fx/fy)
#     v_rot = a*(y*y/fy+fy) - b*x*y/fx - g*x*(fy/fx)
    
#     # Reference:
#     # # Equations 1 and 2 from Passive Navigation as a Pattern Recognition Problem -- but with fx and fy   
#     # x, y = x/fx, y/fy
#     # u_rot = a*(x*y*fx) - b*(x*x+ 1)*fx + g*y*fx
#     # v_rot = a*(y*y+1)*fy - b*x*y*fy - g*x*fy
    
#     return np.dstack((u_rot, v_rot))


# def u_trans(v_x, v_y, v_z, depth, intrinsics):
#     u = v_x
#     v = v_y
#     w = v_z
#     Z = depth

    
#     fx, fy = intrinsics[0,0], intrinsics[1,1]
#     cx, cy = intrinsics[0,2], intrinsics[1,2]
#     grid_size = depth.shape
#     (x, y) = np.meshgrid(np.arange(0, grid_size[1], dtype=np.float32) - (cx + 0.5),
#                          np.arange(0, grid_size[0], dtype=np.float32) - (cy + 0.5)
#                          )
#     x, y = x/fx, y/fy
#     # Equations 1 and 2 from page 3 of Passive Navigation as a Pattern Recognition Problem
#     # Substitute x_0 and y_0
#     u_t = (w * x - fx* u ) / Z
#     v_t = (w * y - fy* v )  / Z

#     return np.dstack((u_t, v_t))

# def compute_flow(v_x, v_y, v_z, yaw, pitch, roll, depth, intrinsics):
#     return u_rot(yaw, pitch, roll, intrinsics, depth.shape) + u_trans(v_x, v_y, v_z, depth, intrinsics)

# ################################################
# def discrete_optical_flow_due_to_rotation(yaw, pitch, roll, focal_length, grid_size):
#     a = pitch
#     b = yaw
#     g = roll
#     f = focal_length

#     (x, y) = np.meshgrid(np.arange(0, grid_size[1], dtype=np.float32) - grid_size[1]/2.0 + 0.5,
#                          np.arange(0, grid_size[0], dtype=np.float32) - grid_size[0]/2.0 + 0.5)

#     # Equations 1 and 2 from page 3 of Passive Navigation as a Pattern Recognition Problem
#     u_rot = a*x*y/f - b*(x*x/f + f) + g*y
#     v_rot = a*(y*y/f+f) - b*x*y/f - g*x

#     return np.dstack((u_rot, v_rot))

# def discrete_optical_flow_due_to_translation(v_x, v_y, v_z, depth, focal_length):
#     u = v_x
#     v = v_y
#     w = v_z
#     Z = depth
#     f = focal_length

#     grid_size = depth.shape
#     (x, y) = np.meshgrid(np.arange(0, grid_size[1], dtype=np.float32) - grid_size[1]/2.0 + 0.5,
#                          np.arange(0, grid_size[0], dtype=np.float32) - grid_size[0]/2.0 + 0.5)

#     # Equations 1 and 2 from page 3 of Passive Navigation as a Pattern Recognition Problem
#     # Substitute x_0 and y_0
#     u_t = (w * x - u * f) / Z
#     v_t = (w * y - v * f) / Z

#     return np.dstack((u_t, v_t))
    
# def discrete_optical_flow(v_x, v_y, v_z, yaw, pitch, roll,
#                           depth, focal_length):
#     flow_rot = discrete_optical_flow_due_to_rotation(yaw, pitch, roll, focal_length, depth.shape)
#     flow_t   = discrete_optical_flow_due_to_translation(v_x, v_y, v_z, depth, focal_length)
#     return flow_t + flow_rot

# # Generate an HSV image using color to represent the gradient direction in a optical flow field
def visualize_optical_flow(flow):
    theta = np.mod(np.arctan2(flow[:, :, 0], flow[:, :, 1]) + 2*np.pi, 2*np.pi)

    flow_norms = np.linalg.norm(flow, axis=2)
    flow_norms_normalized = flow_norms / np.max(flow_norms)

    flow_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255 * flow_norms_normalized
    flow_hsv[:, :, 2] = 255 * (flow_norms_normalized > 0)

    return flow_hsv


# # Sometimes need to convert from NED to image coordinates and project to x, y position in image
# def cvt_ned_image(axis):
#     return np.array((axis[1], axis[2], axis[0]))

# def cvt_image_ned(axis):
#     return np.array((axis[2], axis[0], axis[1]))

# def body_rates_from_quaternions(q_start, q_end, delta):
#     r_start = Rotation.from_quat(q_start)
#     r_end   = Rotation.from_quat(q_end)
#     r_diff = r_start.inv() * r_end
#     return r_diff.as_euler('zyx') / delta




###############################################################################################################################################