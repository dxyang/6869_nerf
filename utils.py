import math
import numpy as np

def check_pose_error(That_A_B: np.array, T_A_B: np.array):
    T_Ahat_A = np.matmul(That_A_B, np.linalg.inv(T_A_B))
    rotation_error = np.arccos(
        (np.trace(T_Ahat_A[:3, :3]) - 1.0) / 2.0
    ) * 180.0 / np.pi
    translation_error = np.linalg.norm(T_Ahat_A[:3, 3])
    return translation_error, rotation_error


def sample_unit_sphere():
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/
    theta = 2 * np.pi * np.random.rand()
    phi = math.acos(1 - 2 * np.random.rand())
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)

    return np.array([x, y, z])


def quat_from_axis_angle(axis: np.array, angle_rads: float):
    # don't divide by zero
    axis = np.squeeze(axis)
    length = np.linalg.norm(axis)
    if length == 0:
        if angle_rads == 0:
            assert False
        else:
            return np.array([0.0, 0.0, 0.0, 1.0]) #xyzw

    normalized_axis = axis / length
    sin_half_angle = math.sin(angle_rads * 0.5)

    return np.array([
        normalized_axis[0] * sin_half_angle,
        normalized_axis[1] * sin_half_angle,
        normalized_axis[2] * sin_half_angle,
        math.cos(angle_rads * 0.5),
    ])


def rotation_matrix_from_quat(q: np.array):
    q = q.squeeze() # xyzw
    assert np.isclose(np.linalg.norm(q) ** 2, 1.0) # quat should be normalized?

    tx = q[0] + q[0]
    ty = q[1] + q[1]
    tz = q[2] + q[2]

    twx = q[3] * tx
    twy = q[3] * ty
    twz = q[3] * tz

    txx = q[0] * tx
    txy = q[0] * ty
    txz = q[0] * tz

    tyy = q[1] * ty
    tyz = q[1] * tz
    tzz = q[2] * tz

    mat3 = np.zeros((3, 3))
    mat3[0][0] = 1 - (tyy + tzz)
    mat3[0][1] = txy - twz
    mat3[0][2] = txz + twy
    mat3[1][0] = txy + twz
    mat3[1][1] = 1 - (txx + tzz)
    mat3[1][2] = tyz - twx
    mat3[2][0] = txz - twy
    mat3[2][1] = tyz + twx
    mat3[2][2] = 1 - (txx + tyy)

    assert(is_rotation_matrix(mat3))

    return mat3


def is_rotation_matrix(mat3: np.array):

    if np.linalg.det(mat3) > 0:
        dist = np.linalg.norm(np.dot(np.array(mat3).T, mat3) - np.eye(3))
    else:
        dist = 1e+9

    return abs(dist) <= 1e-3


def rotation_matrix_from_axis_angle(axis: np.array, angle_rads: float):
    q = quat_from_axis_angle(axis, angle_rads)
    rot = rotation_matrix_from_quat(q)
    return rot
