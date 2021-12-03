import cv2
import torch
import numpy as np


def downsample_dense_flow(flow, scale_factor):
    flow_scaled = cv2.resize(flow, None, fx=scale_factor[1], fy=scale_factor[0])
    return flow_scaled

def dense_flow_to_sparse_flow_list(flow, coordinate_scale=1.0):
    x = np.arange(0, flow.shape[1], 1, dtype=np.float32) * coordinate_scale + coordinate_scale / 2.0
    y = np.arange(0, flow.shape[0], 1, dtype=np.float32) * coordinate_scale + coordinate_scale / 2.0
    xv, yv = np.meshgrid(x, y)

    xv = xv.flatten()
    yv = yv.flatten()
    flow_x = flow[:, :, 0].flatten()
    flow_y = flow[:, :, 1].flatten()

    flow_list = np.stack((xv, yv, flow_x, flow_y), axis=1)

    return flow_list

# Plot a dense flow field on an image
def flow_quiverplot(flow, image=None, scale_factor=(0.05, 0.05), quiver_scale=1.0, color=(255, 0, 0), thickness=1, angular=False):
    if image is None:
        image = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    flow_scaled = downsample_dense_flow(flow, scale_factor)
    flow_list =  dense_flow_to_sparse_flow_list(flow_scaled, coordinate_scale=image.shape[0]/flow_scaled.shape[0])
    return sparse_flow_as_quiver_plot(flow_list, image, quiver_scale, color, thickness, angular)

# Accept a list of optical flow vectors, plot them as arrows overlayed on an image
def sparse_flow_as_quiver_plot(flow_list, image, quiver_scale=1.0, color=(255,0,0), thickness=1, angular=False):
    for flow in flow_list:
        start = flow[0:2]

        if not angular:
            end = flow[0:2] + quiver_scale * flow[2:4]
        else:
            end = flow[0:2] + quiver_scale * flow[2:4] / np.linalg.norm(flow[2:4])

        # print(type(start), type(end))
        start= start.astype(np.uint8)
        end = end.astype(np.uint8)
        cv2.line(image, tuple(start), tuple(end), color, thickness)

        angle = np.arctan2(flow[2], flow[3])

        angle_left = angle + np.pi / 4
        angle_right = angle - np.pi / 4

        if not angular:
            mag = quiver_scale * np.linalg.norm(flow[2:4])
        else:
            mag = quiver_scale

        tip_left_head  = end - 0.5 * mag * np.array((np.sin(angle + np.pi/4), np.cos(angle + np.pi/4)))
        tip_right_head = end - 0.5 * mag * np.array((np.sin(angle - np.pi/4), np.cos(angle - np.pi/4)))

        cv2.line(image, tuple(end), tuple(tip_left_head.astype(np.int32)), color, thickness)
        cv2.line(image, tuple(end), tuple(tip_right_head.astype(np.int32)), color, thickness)

    return image


def computedepth(x, device):    
    tr,rot, intrinsics, flow = x

    B,C,H, W = flow.shape
    depth = torch.zeros((B,1,H, W), dtype=torch.float32)
    for b in range(B):
        # obtain focal length
        fx,fy = intrinsics[b][0,0], intrinsics[b][1,1]
        cx, cy = intrinsics[b][0,2], intrinsics[b][1,2]

        ## build the coordinates 
        y_, x_ = torch.meshgrid(torch.arange(0,H), torch.arange(0,W))
        # apply principal point correction 
        X = x_.to(device, dtype=torch.float32) - cx ; Y = y_.to(device, dtype=torch.float32) - cy 

        uxrot = rot[b][0]*(X*Y/fy) - rot[b][1]*(X*X/fx+ fx) + rot[b][2]*Y*(fx/fy)
        uyrot = rot[b][0]*(Y*Y/fy+fy) - rot[b][1]*X*Y/fx - rot[b][2]*X*(fy/fx)

        urot = torch.stack((uxrot,uyrot), dim=0)
        derotated_flow = flow[b] - urot

        # for translation inputs
        uxtrans = (-tr[b][0]*fx + tr[b][2]*X) 
        uytrans = (-tr[b][1]*fy + tr[b][2]*Y) 
        utrans_ = torch.stack((uxtrans,uytrans), dim=0 )        

        Z =  utrans_/derotated_flow
        Z[torch.isinf(Z)] = 0; Z[torch.isnan(Z)] = 0; Z[flow[b]==0]=0; Z[Z<0] = 0; Z[Z>100] = 0
        Z = torch.mean(Z, dim=0)
        # Z = 0.5* torch.sum(Z[0] + Z[1])
        print(Z.shape)
        depth[b] = Z
    return depth


def computeflow(x, device):    
    tr,rot, intrinsics, depth = x

    B,C,H, W = depth.shape
    flow = torch.zeros((B,2,H, W)).to(device, dtype=torch.float32)
    for b in range(B):
        # obtain focal length
        fx,fy = intrinsics[b][0,0], intrinsics[b][1,1]
        cx, cy = intrinsics[b][0,2], intrinsics[b][1,2]

        ## build the coordinates 
        y_, x_ = torch.meshgrid(torch.arange(0,H), torch.arange(0,W))
        # apply principal point correction 
        X = x_.to(device, dtype=torch.float32) - cx ; Y = y_.to(device, dtype=torch.float32) - cy 

        uxrot = rot[b][0]*(X*Y/fy) - rot[b][1]*(X*X/fx+ fx) + rot[b][2]*Y*(fx/fy)
        uyrot = rot[b][0]*(Y*Y/fy+fy) - rot[b][1]*X*Y/fx - rot[b][2]*X*(fy/fx)
        urot = torch.stack((uxrot,uyrot), dim=0)
        
        # for translation inputs
        uxtrans =  (-tr[b][0]*fx + tr[b][2]*X) 
        uytrans =  (-tr[b][1]*fy + tr[b][2]*Y) 
        utrans = (1/depth[b]) * torch.stack((uxtrans,uytrans), dim=0)        
        flow_ = urot + utrans
        flow[b] = flow_
    return flow