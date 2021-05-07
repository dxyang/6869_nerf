'''
Torch implementations adapted from the numpy implementations

Numpy implementations from
https://github.com/NxRLab/ModernRobotics/blob/master/packages/Python/modern_robotics/core.py
'''
import numpy as np
import torch

'''
torch implementations
'''
def screwToMatrixExp4_torch(expc6):
    # convert R6 to axis angle (w, v) and theta
    axisang, theta = AxisAng6_torch(expc6)
    omg, nu = axisang[:3], axisang[3:]
    so3mat = vecToSo3_torch(omg)

    # calculate the rotation exp([w] * theta)
    exp3 = MatrixExp3_torch_v2(omg, theta)
    #exp3 = MatrixExp3_torch(so3mat * theta)

    # calculate the translation K(S, theta)
    KStheta = (torch.eye(3)*theta + (1-torch.cos(theta))*so3mat + (theta-torch.sin(theta))*torch.mm(so3mat, so3mat))
    KStheta = torch.mm(KStheta, torch.unsqueeze(nu,1))

    # exponential SE3 exp([S] * theta)
    expStheta = torch.Tensor(4,4)
    expStheta[:3,:3] = exp3
    expStheta[:3,3] = KStheta.squeeze()
    expStheta[3,3] = 1
    return expStheta


def AxisAng6_torch(expc6):
    """
    Accepts a 6-vector expc6
    Returns a (expc6, theta) 6-vector, scalar tuple
    """
    theta = torch.norm(expc6[:3])
    return (expc6/theta, theta)


def AxisAng3_torch(expc3):
    theta = torch.norm(expc3)
    return (expc3/theta, theta)


def vecToSo3_torch(omg):
    """
    3-vector to so3 3x3 matrix
    """
    return torch.Tensor([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])


def so3ToVec_torch(so3mat):
    return torch.Tensor([so3mat[2][1], so3mat[0][2], so3mat[1][0]])


def MatrixExp3_torch_v2(omg, theta):
    omgmat = vecToSo3_torch(omg)
    return torch.eye(3) + torch.sin(theta)*omgmat + (1 - torch.cos(theta)) * torch.mm(omgmat,omgmat)


def MatrixExp3_torch(so3mat):
    omgtheta = so3ToVec_torch(so3mat)
    theta = AxisAng3_torch(omgtheta)[1]
    omgmat = so3mat / theta
    return torch.eye(3) + torch.sin(theta)*omgmat + (1 - torch.cos(theta)) * torch.mm(omgmat,omgmat)


'''
numpy implementations
'''
def NearZero(z):
    """Determines whether a scalar is small enough to be treated as zero
    :param z: A scalar input to check
    :return: True if z is close to zero, false otherwise
    Example Input:
        z = -1e-7
    Output:
        True
    """
    return abs(z) < 1e-6


def Normalize(V):
    """Normalizes a vector
    :param V: A vector
    :return: A unit vector pointing in the same direction as z
    Example Input:
        V = np.array([1, 2, 3])
    Output:
        np.array([0.26726124, 0.53452248, 0.80178373])
    """
    return V / np.linalg.norm(V)


def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation
    :param omg: A 3-vector
    :return: The skew symmetric representation of omg
    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])


def so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat
    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])


def AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form
    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle
    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (Normalize(expc3), np.linalg.norm(expc3))


def MatrixExp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat
    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    """
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)


def AxisAng6(expc6):
    """Converts a 6-vector of exponential coordinates into screw axis-angle
    form
    :param expc6: A 6-vector of exponential coordinates for rigid-body motion
                  S*theta
    :return S: The corresponding normalized screw axis
    :return theta: The distance traveled along/about S
    Example Input:
        expc6 = np.array([1, 0, 0, 1, 2, 3])
    Output:
        (np.array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 1.0)
    """
    theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
    if NearZero(theta):
        theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
    return (np.array(expc6 / theta), theta)


# useful for check/asserts/sanity checks
def DistanceToSO3(mat):
    """Returns the Frobenius norm to describe the distance of mat from the
    SO(3) manifold
    :param mat: A 3x3 matrix
    :return: A quantity describing the distance of mat from the SO(3)
             manifold
    Computes the distance from mat to the SO(3) manifold using the following
    method:
    If det(mat) <= 0, return a large number.
    If det(mat) > 0, return norm(mat^T.mat - I).
    Example Input:
        mat = np.array([[ 1.0,  0.0,   0.0 ],
                        [ 0.0,  0.1,  -0.95],
                        [ 0.0,  1.0,   0.1 ]])
    Output:
        0.08835
    """
    if np.linalg.det(mat) > 0:
        return np.linalg.norm(np.dot(np.array(mat).T, mat) - np.eye(3))
    else:
        return 1e+9


def TestIfSO3(mat):
    """Returns true if mat is close to or on the manifold SO(3)
    :param mat: A 3x3 matrix
    :return: True if mat is very close to or in SO(3), false otherwise
    Computes the distance d from mat to the SO(3) manifold using the
    following method:
    If det(mat) <= 0, d = a large number.
    If det(mat) > 0, d = norm(mat^T.mat - I).
    If d is close to zero, return true. Otherwise, return false.
    Example Input:
        mat = np.array([[1.0, 0.0,  0.0 ],
                        [0.0, 0.1, -0.95],
                        [0.0, 1.0,  0.1 ]])
    Output:
        False
    """
    return abs(DistanceToSO3(mat)) < 1e-3
