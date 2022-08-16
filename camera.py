import numpy as np
import cv2
import copy
from calibration import generate_truth_locations, recover_homographies, recover_intrinsics


class Camera:
    def __init__(self) -> None:
        pass

    @staticmethod
    def static_calibrate(gray_images: np.ndarray, chessboard_size, chessboard_cell_size, save_image=True, save_dir='./images/') -> np.ndarray:
        # H矩阵
        H_s = []
        for i in range(gray_images.shape[0]):
            # 寻找棋盘格角点的模式 自适应阈值+标准化图片
            chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

            ret, corners = cv2.findChessboardCorners(
                gray_images[i], chessboard_size, chessboard_flags)

            # 若未找到角点则退出
            if ret is not True:
                continue

            # 进一步优化角点坐标
            criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                        cv2.TERM_CRITERIA_EPS, 30, 0.001)
            corners = cv2.cornerSubPix(
                gray_images[i], corners, (11, 11), (-1, -1), criteria)
            # 保存棋盘格角点图片
            if save_image:
                img = cv2.drawChessboardCorners(copy.deepcopy(
                    gray_images[i]), chessboard_size, corners, ret,)
                for j in range(3):
                    cv2.circle(img, (int(corners[j][0][0]), int(
                        corners[j][0][1])), 60, (0, 0, 255), 0)
            cv2.imwrite(save_dir+'corners_{}.jpg'.format(i), img=img)

            # 生成角点在世界坐标系下的坐标点
            truth_locations = generate_truth_locations(
                chessboard_size, chessboard_cell_size)

            # 计算H矩阵
            corners = corners.reshape(-1, 2)
            H = recover_homographies(corners, truth_locations)
            H /= H[-1, -1]
            H_s.append(H)
        H_s = np.array(H_s)
        K = recover_intrinsics(H_s)
        return K
