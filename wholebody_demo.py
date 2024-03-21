import time

import cv2

from rtmlib import PoseTracker, Wholebody, draw_skeleton, Body


# import numpy as np

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture('./2.mp4')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = PoseTracker(Body,
                        det_frequency=10,
                        to_openpose=openpose_skeleton,
                        mode='performance',
                        backend=backend,
                        device=device)

frame_idx = 0
img = cv2.imread('./demo.jpg')
keypoints, scores = wholebody(img)

img_show = draw_skeleton(img,
                         keypoints,
                         scores,
                         openpose_skeleton=openpose_skeleton,
                         kpt_thr=0.43)
cv2.imshow('img', img_show)
cv2.waitKey()

#
# while cap.isOpened():
#     success, frame = cap.read()
#     frame_idx += 1
#
#     if not success:
#         break
#     s = time.time()
#
#     det_time = time.time() - s
#     print('det: ', det_time)
#
#     img_show = frame.copy()
#
#     # if you want to use black background instead of original image,
#     img_show = np.zeros(img_show.shape, dtype=np.uint8)
#

#
#     img_show = cv2.resize(img_show, (960, 540))
#     cv2.imshow('img', img_show)
#     cv2.waitKey(10)
