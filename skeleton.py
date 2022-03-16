# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import numpy as np
import torch
import math


def rotation_matrix(axis, theta):
    """
    Code modified from the original https://github.com/lshiwjx/2s-AGCN/blob/master/data_gen/rotation.py#L5
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """
    Code modified from the original https://github.com/lshiwjx/2s-AGCN/blob/master/data_gen/rotation.py#L24
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Code modified from the original https://github.com/lshiwjx/2s-AGCN/blob/master/data_gen/rotation.py#L28
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_h36m_joint_names():
    return [
        'hip',  # 0
        'left_hip',  # 1
        'left_knee',  # 2
        'left_ankle',  # 3
        'right_hip',  # 4
        'right_knee',  # 5
        'right_ankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'left_shoulder',  # 11
        'left_elbow',  # 12
        'left_wrist',  # 13
        'right_shoulder',  # 14
        'right_elbow',  # 15
        'right_wrist',  # 16
    ]


def get_dope_joint_names():
    return [
        'right_ankle',  # 0
        'left_ankle',  # 1
        'right_knee',  # 2
        'left_knee',  # 3
        'right_hip',  # 4
        'left_hip',  # 5
        'right_wrist',  # 6
        'left_wrist',  # 7
        'right_elbow',  # 8
        'left_elbow',  # 9
        'right_shoulder',  # 10
        'left_shoulder',  # 11
        'headtop',  # 12
    ]


def get_h36m_skeleton():
    return np.array(
        [
            [
                # right
                [0, 4],
                [4, 5],
                [5, 6],
                [0, 7],
                [7, 8],
                [8, 9],
                [9, 10],
                [8, 14],
                [14, 15],
                [15, 16]
            ],
            # left
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [8, 11],
                [11, 12],
                [12, 13],
            ]
        ]
    )


def convert_jts(jts, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()
    list_out = []
    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            idx = src_names.index(jn)
            list_out.append(jts[:, idx])
        else:
            if src == 'dope' and dst == 'h36m':
                if jn == 'hip':
                    idx = [src_names.index('left_hip'), src_names.index('right_hip')]
                    list_out.append(jts[:, idx].mean(1))
                elif jn == 'Spine (H36M)':
                    idx = [src_names.index('left_hip'), src_names.index('right_hip'),
                           src_names.index('left_shoulder'), src_names.index('right_shoulder')
                           ]
                    list_out.append(jts[:, idx].mean(1))
                elif jn == 'neck':
                    idx = [src_names.index('left_shoulder'), src_names.index('right_shoulder')]
                    list_out.append(jts[:, idx].mean(1))
                elif jn == 'Head (H36M)':
                    idx = [src_names.index('headtop'), src_names.index('left_shoulder'),
                           src_names.index('right_shoulder')]
                    list_out.append(jts[:, idx].mean(1))
            else:
                import ipdb
                ipdb.set_trace()
    out = np.stack(list_out, 1)
    return out


def get_h36m_traversal():
    # bottom left/right
    traversal_bottom_left = ['left_hip', 'left_knee', 'left_ankle']
    parents_bottom_left = ['hip', 'left_hip', 'left_knee']
    traversal_bottom_right = ['right_hip', 'right_knee', 'right_ankle']
    parents_bottom_right = ['hip', 'right_hip', 'right_knee']

    # top left/right
    traversal_top_left = ['Spine (H36M)', 'neck', 'left_shoulder', 'left_elbow', 'left_wrist', 'Head (H36M)', 'headtop']
    parents_top_left = ['hip', 'Spine (H36M)', 'neck', 'left_shoulder', 'left_elbow', 'neck', 'Head (H36M)']
    traversal_top_right = ['right_shoulder', 'right_elbow', 'right_wrist']
    parents_top_right = ['neck', 'right_shoulder', 'right_elbow']

    traversal = traversal_bottom_left + traversal_bottom_right + traversal_top_left + traversal_top_right
    parents = parents_bottom_left + parents_bottom_right + parents_top_left + parents_top_right

    names = get_h36m_joint_names()
    traversal_idx = []
    parents_idx = []
    for i in range(len(traversal)):
        traversal_idx.append(names.index(traversal[i]))
        parents_idx.append(names.index(parents[i]))

    assert len(traversal_idx) == len(parents_idx)

    return traversal_idx, parents_idx


def preprocess_skeleton(pose, center_joint=[0], xaxis=[1, 4], yaxis=[7, 0], iter=5, sanity_check=True,
                        norm_x_axis=True, norm_y_axis=True):
    """
    Code modified from the original https://github.com/lshiwjx/2s-AGCN/blob/master/data_gen/preprocess.py#L8
    Preprocess skeleton such that we disentangle the root orientation and the relative pose
    Default values are for h36m_plus skeleton (center=hip, xaxis=left_shoulder/right_shoulder, yaxis=spine/hip
    Args:
        - pose: [t,k,3] np.array
        - center_joint: list
        - xaxis: list
        - yaxis: list
        - iter: int
    Return:
        - pose_rel: [t,k,3] np.array
        - pose_center: [t,3] np.array
        - matrix: [t,3,3] np.array
    """
    pose_rel = pose.copy()

    # Sub the center joint (pelvis 17)
    pose_center = pose_rel[:, center_joint].mean(1, keepdims=True)
    pose_rel = pose_rel - pose_center

    list_matrix = []
    list_diff = []
    for t in range(pose_rel.shape[0]):

        matrix = []
        inv_matrix = []
        for _ in range(iter):
            # parallel the bone between hip(jpt 0) and spine(jpt 7) to the Y axis
            if norm_y_axis:
                joint_bottom = pose_rel[t, yaxis[0]]
                joint_top = pose_rel[t, yaxis[1]]
                axis = np.cross(joint_top - joint_bottom, [0, 1, 0]).astype(np.float32)
                angle = angle_between(joint_top - joint_bottom, [0, 1, 0]).astype(np.float32)
                matrix_x = rotation_matrix(axis, angle).astype(np.float32)
                pose_rel[t] = (matrix_x.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
                matrix.append(matrix_x)

            # parallel the bone between right_shoulder(jpt 0) and left_shoulder(jpt 7) to the X axis
            if norm_x_axis:
                joint_rshoulder = pose_rel[t, xaxis[0]]
                joint_lshoulder = pose_rel[t, xaxis[1]]
                axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0]).astype(np.float32)
                angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0]).astype(np.float32)
                matrix_y = rotation_matrix(axis, angle).astype(np.float32)
                pose_rel[t] = (matrix_y.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
                matrix.append(matrix_y)

        # compute the center orient rotmat
        matrix.reverse()
        mat = matrix[0]
        for x in matrix[1:]:
            mat = mat @ x
        list_matrix.append(mat)

        if sanity_check:
            # sanity check for computing the inverse matrix step by step
            matrix.reverse()
            inv_mat = np.linalg.inv(matrix[0])
            for x in matrix[1:]:
                inv_mat = inv_mat @ np.linalg.inv(x)
            pose_centered_t_bis = (inv_mat.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
            pose_centered_t = pose[t] - pose_center[t]
            err = np.abs(pose_centered_t_bis - pose_centered_t).sum()
            # print(err)
            assert err < 1e-5
            inv_matrix.append(inv_mat)

            # sanity check for matrix multiplication
            pose_rel_bis = pose.copy() - pose_center
            pose_rel_t_bis = (mat.reshape(1, 3, 3) @ pose_rel_bis[t].reshape(-1, 3, 1)).reshape(-1, 3)
            err = np.abs(pose_rel_t_bis - pose_rel[t]).sum()
            # print(err)
            assert err < 1e-5

            # inv bis
            inv_mat_bis = np.linalg.inv(mat)
            pose_centered_t_bis_bis = (inv_mat_bis.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
            err = np.abs(pose_centered_t_bis_bis - pose_centered_t).sum()
            # print(err)
            assert err < 1e-5

    orient_center = np.stack(list_matrix)
    return pose_rel, pose_center.reshape(-1, 3), orient_center


def normalize_skeleton_by_bone_length(x, y, traversal, parents):
    """
    Args:
        - pred: [k,3]
        - gt: [k,3]
        - traversal: list of len==k
        - parents: list of len==k
    """
    x_norm = x.copy()

    for i in range(len(traversal)):
        i_joint = traversal[i]
        i_parent = parents[i]
        y_len = np.linalg.norm(y[i_joint] - y[i_parent])
        x_vec = x[i_joint] - x[i_parent]
        x_len = np.linalg.norm(x_vec)
        # import ipdb
        # ipdb.set_trace()
        if x_len > 0:
            x_norm[i_joint] = x_norm[i_parent] + x_vec * y_len / x_len
    return x_norm
