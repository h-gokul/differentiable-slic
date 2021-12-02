import numpy as np
import copy
import cv2

def forwardflow_warp_(im, flow_):
    flow=flow_.copy()
    """
    Use optical flow to warp image to the next
    :param im: image to warp
    :param flow: optical flow
    :return: warped image
    """
    from scipy import interpolate
    image_height = im.shape[0]
    image_width = im.shape[1]
    flow_height = flow.shape[0]
    flow_width = flow.shape[1]
    n = image_height * image_width

    # prev image's coords grid
    (iy1, ix1) = np.mgrid[0:image_height, 0:image_width]
    
    # compute ix2 = x1+du, iy2 = y1+dv next image's coord grid     
    (iy2, ix2) = np.mgrid[0:flow_height, 0:flow_width]
    ix2 = ix2.astype(np.float64)
    iy2 = iy2.astype(np.float64)
    ix2 += flow[:,:,0]
    iy2 += flow[:,:,1]

    # compute a mask that indicates coord-values(ix2, iy2) values going outside input shape
    mask = np.logical_or(ix2 <0 , ix2 > flow_width)
    mask = np.logical_or(mask, iy2 < 0)
    mask = np.logical_or(mask, iy2 > flow_height)
    
    # clip coord-values of (ix2, iy2) values going outside input image shape
    ix2 = np.minimum(np.maximum(ix2, 0), flow_width)
    iy2 = np.minimum(np.maximum(iy2, 0), flow_height)

    # flatten xgrid, ygrid and concatenate 
    points1 = np.concatenate((ix1.reshape(n,1), iy1.reshape(n,1)), axis=1)
    points2 = np.concatenate((ix2.reshape(n,1), iy2.reshape(n,1)), axis=1)
    
    warp = np.zeros((image_height, image_width, im.shape[2]))
    for i in range(im.shape[2]):
        channel = im[:, :, i] # get the pixel values in the image
        
        values = channel.reshape(n, 1) # reshape to a 1D vector
        # place and interpolate the pixel values in points2 vector, from points1 vector
        new_channel = interpolate.griddata(points1, values, points2, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width]) # reshape back to image channel
                
        new_channel[mask] = 1 # apply the black region mask
        warp[:, :, i] = new_channel.astype(np.uint8)

    return warp.astype(np.uint8)

def forwardflow_warp(im0, flow):
    opt_flow = flow.copy()
    h, w = im0.shape[:2]
    opt_flow[:,:,0] += np.arange(w)
    opt_flow[:,:,1] += np.arange(h)[:,np.newaxis]
    im1_out = cv2.remap(im0, opt_flow, None, cv2.INTER_LINEAR)
    return im1_out


def compute_PlanarDepth(perspective_depth, f):
    h = perspective_depth.shape[0]
    w = perspective_depth.shape[1]
    i_c = float(h) / 2 - 1
    j_c = float(w) / 2 - 1
    columns, rows = np.meshgrid(np.linspace(0, w-1, num=w), np.linspace(0, h-1, num=h))
    distance_from_center = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
    planar_depth = perspective_depth / (1 + (distance_from_center / f)**2)**(0.5)
    return planar_depth

class CameraBase(object):
    def __init__(self, focal, imageSize):
        self.focal = focal
        self.imageSize = copy.deepcopy(imageSize) # List or tuple, (height, width)
        self.size = self.imageSize[0] * self.imageSize[1]

        self.pu = self.imageSize[1] / 2
        self.pv = self.imageSize[0] / 2

        self.cameraMatrix = np.eye(3, dtype = np.float)
        self.cameraMatrix[0, 0] = self.focal
        self.cameraMatrix[1, 1] = self.focal
        self.cameraMatrix[0, 2] = self.pu
        self.cameraMatrix[1, 2] = self.pv

        self.worldR = np.zeros((3,3), dtype = np.float)
        self.worldR[0, 1] = 1.0
        self.worldR[1, 2] = 1.0
        self.worldR[2, 0] = 1.0

        self.worldRI = np.zeros((3,3), dtype = np.float)
        self.worldRI[0, 2] = 1.0
        self.worldRI[1, 0] = 1.0
        self.worldRI[2, 1] = 1.0

    def from_camera_frame_to_image(self, coor):
        """
        coor: A numpy column vector, 3x1.
        return: A numpy column vector, 2x1.
        """
        
        # coor = self.worldR.dot(coor)
        x = self.cameraMatrix.dot(coor)
        x = x / x[2,:]

        return x[0:2, :]

    def from_depth_to_x_y(self, depth):
        wIdx = np.linspace( 0, self.imageSize[1] - 1, self.imageSize[1] )
        hIdx = np.linspace( 0, self.imageSize[0] - 1, self.imageSize[0] )

        u, v = np.meshgrid(wIdx, hIdx)

        u = u.astype(np.float)
        v = v.astype(np.float)
        
        x = ( u - self.pu ) * depth / self.focal
        y = ( v - self.pv ) * depth / self.focal

        coor = np.zeros((3, self.size), dtype = np.float)
        coor[0, :] = x.reshape((1, -1))
        coor[1, :] = y.reshape((1, -1))
        coor[2, :] = depth.reshape((1, -1))

        # coor = self.worldRI.dot(coor)

        return coor


def du_dv(nu, nv, imageSize):
    wIdx = np.linspace( 0, imageSize[1] - 1, imageSize[1] )
    hIdx = np.linspace( 0, imageSize[0] - 1, imageSize[0] )
    u, v = np.meshgrid(wIdx, hIdx)
    return nu - u, nv - v


def from_quaternion_to_rotation_matrix(q):
    """
    q: A numpy vector, 4x1.
    """

    qi2 = q[0, 0]**2; qj2 = q[1, 0]**2; qk2 = q[2, 0]**2

    qij = q[0, 0] * q[1, 0]; qjk = q[1, 0] * q[2, 0]; qki = q[2, 0] * q[0, 0]

    qri = q[3, 0] * q[0, 0]; qrj = q[3, 0] * q[1, 0]; qrk = q[3, 0] * q[2, 0]

    s = 1.0 / ( q[3, 0]**2 + qi2 + qj2 + qk2 )
    ss = 2 * s

    R = [\
        [ 1.0 - ss * (qj2 + qk2), ss * (qij - qrk), ss * (qki + qrj) ],\
        [ ss * (qij + qrk), 1.0 - ss * (qi2 + qk2), ss * (qjk - qri) ],\
        [ ss * (qki - qrj), ss * (qjk + qri), 1.0 - ss * (qi2 + qj2) ],\
        ]
    return np.array(R, dtype = np.float)

def read_abspose(pose_list, idx):
    data    = pose_list[idx, :].reshape((-1, 1))
    t = data[:3, 0].reshape((-1, 1))
    q = data[3:, 0].reshape((-1, 1))
    R = from_quaternion_to_rotation_matrix(q)

    return (R.transpose(), -R.transpose().dot(t), q)

def read_absolutepose(data):
    t = data[:3, 0].reshape((-1, 1))
    q = data[3:, 0].reshape((-1, 1))
    R = from_quaternion_to_rotation_matrix(q)
    # print(q, q.shape, t, t.shape)
    return (R.transpose(), -R.transpose().dot(t), q)


def computeflow(depth_0, depth_1, pose0, pose1, K):
    pose0 = read_absolutepose(pose0.reshape((-1, 1)))
    pose1 = read_absolutepose(pose1.reshape((-1, 1)))

    h,w = depth_0.shape
    cam_0 = CameraBase(focal = K[0,0], imageSize = (h,w))

    flow = flow_from_depth(depth_0, depth_1, pose0, pose1, cam_0)
    return flow

def flow_from_depth(depth_0, depth_1, pose0, pose1, cam_0):

    cam_1 = cam_0

    R0, t0, q0 = pose0; R1, t1, q1 = pose1 
    R0Inv = np.linalg.inv(R0); R1Inv = np.linalg.inv(R1)
    # Compute the rotation between the two camera poses.
    R = np.matmul( R1, R0Inv )


    # Calculate the coordinates in the first camera's frame.
    X0C = cam_0.from_depth_to_x_y(depth_0) # Coordinates in the camera frame. z-axis pointing forwards.
    X0  = cam_0.worldRI.dot(X0C)           # Corrdinates in the NED frame. z-axis pointing downwards.
    # The coordinates in the world frame.
    XWorld_0  = R0Inv.dot(X0 - t0)


    # Calculate the coordinates in the second camera's frame.
    X1C = cam_1.from_depth_to_x_y(depth_1) # Coordinates in the camera frame. z-axis pointing forwards.
    X1  = cam_1.worldRI.dot(X1C)           # Corrdinates in the NED frame. z-axis pointing downwards.
    # The coordiantes in the world frame.
    XWorld_1 = R1Inv.dot( X1 - t1 )


    # ====================================
    # The coordinate of the pixels of the first camera projected in the second camera's frame (NED).
    X_01 = R1.dot(XWorld_0) + t1

    # The image coordinates in the second camera.
    X_01C = cam_0.worldR.dot(X_01)                  # Camera frame, z-axis pointing forwards.
    c     = cam_0.from_camera_frame_to_image(X_01C) # Image plane coordinates.

    # Get new u anv v
    u = c[0, :].reshape(cam_0.imageSize)
    v = c[1, :].reshape(cam_0.imageSize)


    # Get the du and dv.
    du, dv = du_dv(u, v, cam_0.imageSize)

    dudv = np.zeros( ( cam_0.imageSize[0], cam_0.imageSize[1], 2), dtype = np.float32 )
    dudv[:, :, 0] = du
    dudv[:, :, 1] = dv
    
    return dudv
    
###########################################################
# def warp_image(im, flow_):
#     flow=flow_.copy()
#     """
#     Use optical flow to warp image to the next
#     :param im: image to warp
#     :param flow: optical flow
#     :return: warped image
#     """
#     from scipy import interpolate
#     image_height = im.shape[0]
#     image_width = im.shape[1]
#     flow_height = flow.shape[0]
#     flow_width = flow.shape[1]
#     n = image_height * image_width

#     # prev image's coords grid
#     (iy1, ix1) = np.mgrid[0:image_height, 0:image_width]
    
#     # compute ix2 = x1+du, iy2 = y1+dv next image's coord grid     
#     (iy2, ix2) = np.mgrid[0:flow_height, 0:flow_width]
#     ix2 = ix2.astype(np.float64)
#     iy2 = iy2.astype(np.float64)
#     ix2 += flow[:,:,0]
#     iy2 += flow[:,:,1]

#     # compute a mask that indicates coord-values(ix2, iy2) values going outside input shape
#     mask = np.logical_or(ix2 <0 , ix2 > flow_width)
#     mask = np.logical_or(mask, iy2 < 0)
#     mask = np.logical_or(mask, iy2 > flow_height)
    
#     # clip coord-values of (ix2, iy2) values going outside input image shape
#     ix2 = np.minimum(np.maximum(ix2, 0), flow_width)
#     iy2 = np.minimum(np.maximum(iy2, 0), flow_height)

#     # flatten xgrid, ygrid and concatenate 
#     points1 = np.concatenate((ix1.reshape(n,1), iy1.reshape(n,1)), axis=1)
#     points2 = np.concatenate((ix2.reshape(n,1), iy2.reshape(n,1)), axis=1)
    
#     warp = np.zeros((image_height, image_width, im.shape[2]))
#     for i in range(im.shape[2]):
#         channel = im[:, :, i] # get the pixel values in the image
        
#         values = channel.reshape(n, 1) # reshape to a 1D vector
#         # place and interpolate the pixel values in points2 vector, from points1 vector
#         new_channel = interpolate.griddata(points1, values, points2, method='cubic')
#         new_channel = np.reshape(new_channel, [flow_height, flow_width]) # reshape back to image channel
        
#         # f, ax =  plt.subplots(1,2,figsize= (10,10))
#         # ax[0].imshow(channel, cmap='gray')
#         # ax[1].imshow(new_channel.astype(np.uint8), cmap='gray')
        
#         new_channel[mask] = 1 # apply the black region mask
#         warp[:, :, i] = new_channel.astype(np.uint8)

#     return warp.astype(np.uint8)