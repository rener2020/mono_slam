from unittest import removeResult
import cv2
import numpy as np
from .orb import orb_detector


def resolve_essential_matrix(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K)
    return E


def decompose_E(E: np.ndarray):
    W = np.array(
        [[0, -1, 0],
         [1, 0, 0],
         [0, 0, 1]]
    )
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U.T[-1]
    return R1, R2, t


def generate_G(R, t, K):
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    G = K@T
    return G


def triangulate(matched_points, G1, G2):
    p3ds = np.zeros((len(matched_points), 3))
    for i in range(len(matched_points)):
        point1 = matched_points[i][0]
        point2 = matched_points[i][1]
        A = np.zeros((4, 4))
        A[0] = point1[0] * G1[2] - G1[0]
        A[1] = point1[1] * G1[2] - G1[1]
        A[2] = point2[0] * G2[2] - G2[0]
        A[3] = point2[1] * G2[2] - G2[1]
        U, S, Vt = np.linalg.svd(A)
        p4d = Vt[-1]
        p3ds[i] = (p4d / p4d[-1])[:3].T
    return p3ds


def check_Rt(R, t, K, matched_points):
    # 第一个相机的位姿
    R1 = np.eye(3)
    t1 = np.zeros((3))
    G1 = generate_G(R1, t1, K)
    # 第二个相机的位姿
    G2 = generate_G(R, t, K)
    # 三角测量算出匹配点在第一个相机下的世界坐标
    p3ds1 = triangulate(matched_points, G1, G2)
    # 计算匹配点在第二个相机下的世界坐标
    p3ds2 = (R @ p3ds1.T).T + t

    # 第一个相机下的匹配点世界坐标深度大于个数
    first_good_num = (p3ds1[:, -1] < 0).sum()
    # 第二个相机下的匹配点世界坐标深度是否大部分都大于0
    second_good_num = (p3ds2[:, -1] < 0).sum()

    return (first_good_num, second_good_num), p3ds1


def resolve_T(gray1, gray2, K):
    matched_points = orb_detector.cross_detect(gray1, gray2)
    E = resolve_essential_matrix(matched_points[:, 0], matched_points[:, 1], K)
    R1, R2, t = decompose_E(E)
    Rs = [R1, R2]
    ts = [t, -t]
    bad_sum = []
    ps = []
    for R in Rs:
        for t in ts:
            bad_points, p = check_Rt(R, t, K, matched_points)
            bad_sum.append(bad_points[0] + bad_points[1])
            ps.append(p)
    index = bad_sum.index(min(bad_sum))
    R, t = Rs[index // 2], ts[index % 2]
    ps = ps[index]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = t
    return T, ps
