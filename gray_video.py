import cv2
import numpy as np


def get_gray_video(video_path):

    # slam 视频
    cap = cv2.VideoCapture(video_path)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gray_video = np.zeros((frame_num, h_, w_), dtype=np.uint8)
    int(cap.get(7))
    count = 0
    for i in range(frame_num):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_video[count] = gray
        count += 1
    return gray_video

gray_video = get_gray_video("./images/3.mp4")

# gray_video2 = get_gray_video("./images/2.mp4")