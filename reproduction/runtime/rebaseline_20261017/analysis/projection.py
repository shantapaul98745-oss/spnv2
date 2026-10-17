"""Projection convention copied from own frozen postprocess; MIT SLAB copyright 2022.
Quaternion [w,x,y,z], pose uses quat2dcm(q).T as in the frozen implementation.
"""
import numpy as np
def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """
    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3), dtype=np.float32)

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def project_keypoints(q, r, K, dist, keypoints):
    """ Projecting 3D keypoints to 2D
        q: quaternion (np.array)
        r: position   (np.array)
        K: camera intrinsic (3,3) (np.array)
        dist: distortion coefficients (5,) (np.array)
        keypoints: N x 3 or 3 x N (np.array)
    """
    # Make sure keypoints are 3 x N
    if keypoints.shape[0] != 3:
        keypoints = np.transpose(keypoints)

    # Keypoints into 4 x N homogenous coordinates
    keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))

    # transformation to image frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    xyz      = np.dot(pose_mat, keypoints) # [3 x N]
    x0, y0   = xyz[0,:] / xyz[2,:], xyz[1,:] / xyz[2,:] # [1 x N] each

    # apply distortion
    r2 = x0*x0 + y0*y0
    cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
    x  = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0)
    y  = y0*cdist + dist[2]*(r2 + 2*y0*y0) + dist[3]*2*x0*y0

    # apply camera matrix
    points2D = np.vstack((K[0,0]*x + K[0,2], K[1,1]*y + K[1,2]))

    return points2D
