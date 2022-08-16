import numpy as np


def svd_solve(A):
    # SVD分解计算最小二乘解
    U, S, V_t = np.linalg.svd(A)
    idx = np.argmin(S)
    least_squares_solution = V_t[idx]

    return least_squares_solution


def generate_truth_locations(chessboard_size: tuple, cell_size: tuple):
    # 生成真实世界棋盘格位置点
    truth_locations = [[[i_x * cell_size[0], i_y * cell_size[1]]
                        for i_x in range(chessboard_size[0])]
                       for i_y in range(chessboard_size[1])]
    truth_locations = np.array(
        truth_locations, dtype=np.float32).reshape(-1, 2)
    return truth_locations


def recover_homographies(corners: np.ndarray, locations: np.ndarray) -> np.ndarray:
    # 构造A矩阵
    A = []  # np.zeros((used_points.shape[0]*2, 9))
    for i in range(corners.shape[0]):
        w_x, w_y = locations[i]
        u, v = corners[i]
        A.append([-w_x, -w_y, -1, 0, 0, 0, u*w_x, u*w_y, u])
        A.append([0, 0, 0, -w_x, -w_y, -1,  v*w_x, v*w_y, v])
    A = np.array(A, dtype=np.float32)
    H = svd_solve(A).reshape((3, 3))
    return H


def generate_v_ij(H_stack, i, j):
    # 构造线性方程组需要的参数矩阵
    M = H_stack.shape[0]

    v_ij = np.zeros((M, 6))
    v_ij[:, 0] = H_stack[:, 0, i] * H_stack[:, 0, j]
    v_ij[:, 1] = H_stack[:, 0, i] * H_stack[:, 1, j] + \
        H_stack[:, 1, i] * H_stack[:, 0, j]
    v_ij[:, 2] = H_stack[:, 1, i] * H_stack[:, 1, j]
    v_ij[:, 3] = H_stack[:, 2, i] * H_stack[:, 0, j] + \
        H_stack[:, 0, i] * H_stack[:, 2, j]
    v_ij[:, 4] = H_stack[:, 2, i] * H_stack[:, 1, j] + \
        H_stack[:, 1, i] * H_stack[:, 2, j]
    v_ij[:, 5] = H_stack[:, 2, i] * H_stack[:, 2, j]

    return v_ij


def recover_intrinsics(homographies):
    # 计算本质矩阵
    M = len(homographies)

    # 重构单应矩阵
    H_stack = np.zeros((M, 3, 3))
    for h, H in enumerate(homographies):
        H_stack[h] = H

    # 约束
    v_00 = generate_v_ij(H_stack, 0, 0)
    v_01 = generate_v_ij(H_stack, 0, 1)
    v_11 = generate_v_ij(H_stack, 1, 1)

    # 约束
    V = np.zeros((2 * M, 6))
    V[:M] = v_01
    V[M:] = v_00 - v_11

    # SVD分解求解最小二乘解
    b = svd_solve(V)

    B0, B1, B2, B3, B4, B5 = b

    # B = K_-T K_-1
    B = np.array([[B0, B1, B3],
                  [B1, B2, B4],
                  [B3, B4, B5]])

    # 求解参数
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / \
        (B[0, 0] * B[1, 1] - B[0, 1] * B[0, 1])
    lambda_ = B[2, 2] - (B[0, 2] * B[0, 2] + v0 *
                         (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    alpha = np.sqrt(lambda_ / B[0, 0])
    beta = np.sqrt(lambda_ * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] * B[0, 1]))
    gamma = -B[0, 1] * alpha * alpha * beta / lambda_
    u0 = gamma * v0 / beta - B[0, 2] * alpha * alpha / lambda_

    # 构建本质矩阵
    K = np.array([[alpha, gamma, u0],
                  [0.,  beta, v0],
                  [0.,    0., 1.]])

    return K
