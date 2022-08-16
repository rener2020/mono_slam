import cv2
import numpy as np


class OrbFeatureDetector:
    # 计算图像特征点计算类

    orb = cv2.ORB_create()

    def __init__(self):
        pass

    def cross_detect(self, gray1: np.ndarray, gray2: np.ndarray):
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)
        # crossCheck=True，两张图的点A→B和B→A各算一次。
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = [[x1]
                        for x1, x2 in matches if x1.distance < 0.75*x2.distance]
        if len(good_matches) <= 8:
            return None, None, None
        # return kp1, kp2, good_matches
        count_matches = len(good_matches)
        matched_points = np.zeros((count_matches, 2, 2),dtype=np.float32)
        for i in range(count_matches):
            matched_points[i][0] = kp1[good_matches[i][0].queryIdx].pt
            matched_points[i][1] = kp2[good_matches[i][0].trainIdx].pt
        return matched_points

orb_detector = OrbFeatureDetector()