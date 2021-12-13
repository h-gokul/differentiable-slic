
import numpy as np
#import cv2
#import pyrr
from scipy.spatial.transform import Rotation as R

def ned2cam(traj):
    '''
    transfer a ned traj to camera frame traj
    '''
    T = np.array([[0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []
    traj_ses = pos_quats2SE_matrices(np.array(traj))

    for tt in traj_ses:
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(SE2pos_quat(ttt))
        
    return np.array(new_traj)

def cam2ned(traj):
    '''
    transfer a camera traj to ned frame traj
    '''
    T = np.array([[0,0,1,0],
                  [1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []
    traj_ses = pos_quats2SE_matrices(np.array(traj))

    for tt in traj_ses:
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(SE2pos_quat(ttt))
        
    return np.array(new_traj)



def line2mat(line_data):
    """
    convert 1x12 to 4x4 pose matrix
    """
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)

def motion2pose(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = line2mat(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose

def pose2motion(data, skip=0):
    """
    Converts all absolute poses (n'x12) to 
    relative poses (nx12) ie n = n'-1
    """
    data_size = data.shape[0]
    all_motion = np.zeros((data_size-1,12))
    for i in range(0,data_size-1-skip):
        pose_curr = line2mat(data[i,:])
        pose_next = line2mat(data[i+1+skip,:])
        motion = pose_curr.I*pose_next
        motion_line = np.array(motion[0:3,:]).reshape(1,12)
        all_motion[i,:] = motion_line
    return all_motion

def abs2rel_pose(pose1, pose2):
    """
    Converts two absolute poses (2x12) to 
    relative poses (1x12) 
    """
    pose_curr = line2mat(pose1)
    pose_next = line2mat(pose2)
    motion = pose_curr.I*pose_next
    motion_1D = np.array(motion[0:3,:]).reshape(1,12)
    return motion_1D


def cam2ned(tr):
    # from xyz to zxy
    return np.array([tr[2], tr[0],tr[1]])
def mat2quat(poses4D, c2n=False):
    outposes=[]
    # convert nx4x4 matrice to  nx7x1 vector with ned datashape
    for pose4D in poses4D:
        rot3x3=pose4D[:3,:3]
        tr=pose4D[:3,3:]
        if c2n:
            tr = cam2ned(tr)
        r_quat = R.from_matrix(rot3x3).as_quat()
        apose = np.vstack((tr,r_quat.reshape(-1,1)))
        outposes.append(apose)
    return np.array(outposes).squeeze()


def aquat_2_reuler(q_start, q_end, degrees = False):
    """obtain relative orientation from absolute quaternion orientation 

    Args:
        q_start ( np.array/list ): first orinetation in quaternions
        q_end (np.array/list): second orientation in quaternions
        delta (frame difference in hz): [description]

    Returns:
        r_diff : relative orientation in euler zyx format
    """

    r_start = R.from_quat(q_start)
    r_end   = R.from_quat(q_end)
    r_diff = r_start.inv() * r_end # last_q-1 @ q

    return r_diff.as_euler('xyz', degrees=degrees)

def matvec2ypr(motion_1x12):               
    """
    Converts a relative pose vector (1x12)
    to rotvec format (1x6)
    """
    motion = np.zeros((6)) 
    pose4D = line2mat(motion_1x12)
    
    motion[:3] = pose4D[0:3, 3:].squeeze()
    motion[3:] = mat2ypr(pose4D[:3, :3])
    return motion
def mat2ypr(rot):
    """
    Converts rotation matrix 3x3 to euler yaw-pitch-roll
    """
    return R.from_dcm(rot).as_euler('zyx')
def mat2pry(rot):
    """
    Converts rotation matrix 3x3 to euler pitch-roll-yaw
    """
    return R.from_dcm(rot).as_euler('xyz')

def matvec2rotvec(motion_1x12):               
    """
    Converts a relative pose vector (1x12)
    to rotvec format (1x6) roll-pitch-yaw
    """
    pose4D = np.matrix(np.eye(4))
    pose4D[0:3,:] = motion_1x12.reshape(3,4)
    motion = SE2se(pose4D)
    return motion
def SE2se(SE_data):
    """
    Converts a relative pose matrix (4x4)
    to euler format (1x6)
    """
    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3,3].T)
    # result[3:6] = mat2pry(SE_data[0:3,0:3]).T
    result[3:6] = SO2so(SE_data[0:3,0:3]).T    
    return result
    
def SO2so(SO_data):
    return R.from_dcm(SO_data).as_rotvec()

def so2SO(so_data):
    return R.from_euler('xyz',so_data).as_matrix()

def se2SE(se_data):
    """
    Convert euler pose vector (N,6) to pose matrix (4x4) 
    """    
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3,0:3] = so2SO(se_data[3:6])
    result_mat[0:3,3]   = np.matrix(se_data[0:3]).T
    return result_mat

def SEs2ses(motion_data):
    """
    Converts all relative poses (nx12) to 
    euler xyz format (nx6)
    """
    data_size = motion_data.shape[0]
    ses = np.zeros((data_size,6))
    for i in range(0,data_size):
        SE = np.matrix(np.eye(4))
        SE[0:3,:] = motion_data[i,:].reshape(3,4)
        ses[i,:] = SE2se(SE)
    return ses

def ses2poses(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = se2SE(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose

def ses2poses_quat(data):
    '''
    convert relative poses nx6 to absolute poses nx7
    '''
    data_size = data.shape[0]
    all_pose_quat = np.zeros((data_size+1,7))
    all_pose_quat[0,:] = np.array([0., 0., 0., 0., 0., 0., 1.])
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = se2SE(data[i,:]) # make vector 4x4 matrix
        pose = pose*data_mat # multiplying relative pose with previous pose
        quat = SO2quat(pose[0:3,0:3]) # convert the pose to quaternion

        # form vector of translation and quaternion rotation
        all_pose_quat[i+1,:3] = np.array([pose[0,3], pose[1,3], pose[2,3]]) 
        all_pose_quat[i+1,3:] = quat    

    return all_pose_quat
    


def so2quat(so_data):
    so_data = np.array(so_data)
    theta = np.sqrt(np.sum(so_data*so_data))
    axis = so_data/theta
    quat=np.zeros(4)
    quat[0:3] = np.sin(theta/2)*axis
    quat[3] = np.cos(theta/2)
    return quat

def quat2so(quat_data):
    quat_data = np.array(quat_data)
    sin_half_theta = np.sqrt(np.sum(quat_data[0:3]*quat_data[0:3]))
    axis = quat_data[0:3]/sin_half_theta
    cos_half_theta = quat_data[3]
    theta = 2*np.arctan2(sin_half_theta,cos_half_theta)
    so = theta*axis
    return so

# input so_datas batch*channel*height*width
# return quat_datas batch*numner*channel
def sos2quats(so_datas,mean_std=[[1],[1]]):
    so_datas = np.array(so_datas)
    so_datas = so_datas.reshape(so_datas.shape[0],so_datas.shape[1],so_datas.shape[2]*so_datas.shape[3])
    so_datas = np.transpose(so_datas,(0,2,1))
    quat_datas = np.zeros((so_datas.shape[0],so_datas.shape[1],4))
    for i_b in range(0,so_datas.shape[0]):
        for i_p in range(0,so_datas.shape[1]):
            so_data = so_datas[i_b,i_p,:]
            quat_data = so2quat(so_data)
            quat_datas[i_b,i_p,:] = quat_data
    return quat_datas

def SO2quat(SO_data):
    rr = R.from_dcm(SO_data)
    return rr.as_quat()

def quat2SO(quat_data):
    return R.from_quat(quat_data).as_dcm()


def pos_quat2SE(quat_data):
    """
    Converts an absolute pose vector (1x7) to 
    flat matrix format (1x12)
    """
    SO = R.from_quat(quat_data[3:7]).as_dcm()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3,:]).reshape(1,12)
    return SE


def pos_quats2SEs(quat_datas):
    """
    Converts all absolute poses (n'x7) to 
    flat matrix format (n'x12)
    """
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len,12))
    for i_data in range(0,data_len):
        SE = pos_quat2SE(quat_datas[i_data,:])
        SEs[i_data,:] = SE
    return SEs


def pos_quats2SE_matrices(quat_datas):
    """
    Converts all absolute poses (n'x7) to 
    matrix format (n'x3 x 4)
    """
    data_len = quat_datas.shape[0]
    SEs = []
    for quat in quat_datas:
        SO = R.from_quat(quat[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3, 0:3] = SO
        SE[0:3, 3]   = quat[0:3]
        SEs.append(SE)
    return SEs


def pos_quats2rot_vec(quat_datas):
    rotvec = []
    data = np.zeros((6))
    for quat in quat_datas:
        rot_vec = R.from_quat(quat[3:7]).as_rotvec()
        data[0:3] = quat[0:3]
        data[3:] = rot_vec
        rotvec.append(data)
    return np.array(rotvec)


def SE2pos_quat(SE_data):
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3,0:3])
    pos_quat[:3] = SE_data[0:3,3].T
    return pos_quat

def pos_rotvec2pos_quat(data):
    pos_quat = np.zeros(7)
    tr = data[:3]; rot = data[3:]

    rot_mat = so2SO(rot) 
    pos_quat[3:] = SO2quat(rot_mat)
    pos_quat[:3] = tr
    return pos_quat


def kitti2tartan(traj, c2n=True):
    '''
    traj: in kitti style, N x 12 numpy array, in camera frame
    output: in TartanAir style, N x 7 numpy array, in NED frame
    '''
    T = np.array([[0,0,1,0],
                  [1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []

    for pose in traj:
        tt = np.eye(4)
        tt[:3,:] = pose.reshape(3,4)
        if c2n:
            ttt=T.dot(tt).dot(T_inv)
        new_traj.append(SE2pos_quat(tt))
        
    return np.array(new_traj)

def tartan2kitti(traj, n2c=True):
    T = np.array([[0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []

    for pose in traj:
        tt = np.eye(4)
        tt[:3,:] = pos_quat2SE(pose).reshape(3,4)
        if n2c:
            ttt=T.dot(tt).dot(T_inv)
            new_traj.append(ttt[:3,:].reshape(12))
        else:
            new_traj.append(tt[:3,:].reshape(12))
        
    return np.array(new_traj)
