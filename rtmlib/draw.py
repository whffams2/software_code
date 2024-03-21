import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .visualization.skeleton import *  # noqa


def draw_bbox(img, bboxes, color=(0, 255, 0)):
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), color, 2)
    return img


def draw_skeleton(img,
                  keypoints,
                  scores,
                  openpose_skeleton=False,
                  kpt_thr=0.5,
                  radius=2,
                  line_width=2):
    if openpose_skeleton:
        skeleton = 'openpose18' if keypoints.shape[1] == 18 else 'openpose134'
    else:
        skeleton = 'coco17' if keypoints.shape[1] == 17 else 'coco133'

    skeleton_dict = eval(f'{skeleton}')
    keypoint_info = skeleton_dict['keypoint_info']
    skeleton_info = skeleton_dict['skeleton_info']

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]

    num_instance = keypoints.shape[0]
    if skeleton.startswith('coco'):
        img = plot_image(img, keypoints, scores)
        # for i in range(num_instance):
        #     img = draw_mmpose(img, keypoints[i], scores[i], keypoint_info,
        #                       skeleton_info, kpt_thr, radius, line_width)

    else:
        for i in range(num_instance):
            img = draw_openpose(img,
                                keypoints[i],
                                scores[i],
                                keypoint_info,
                                skeleton_info,
                                kpt_thr,
                                radius * 2,
                                alpha=0.6,
                                line_width=line_width * 2)
    return img


def plot_image(image, keypoints, score, kpt_thr=0.2):
    # if score < 0.1:
    #     return
    people_line = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [5, 6],
        [5, 11],
        [6, 12],
        [11, 12],
        [5, 7],
        [7, 9],
        [6, 8],
        [8, 10],
        [11, 13],
        [13, 15],
        [12, 14],
        [14, 16]
    ]
    vis_kpt = [s >= kpt_thr for s in score]
    points = keypoints.reshape(17, 2)
    scores = score.reshape(17)
    # import pdb
    # pdb.set_trace()
    for k in range(len(people_line)):

        sign0 = people_line[k][0]
        sign1 = people_line[k][1]
        # import pdb
        # pdb.set_trace()
        if scores[sign0] < 0.2:
            continue
        if scores[sign1] < 0.2:
            continue

        if k < 4:
            cv2.line(image, (int(points[sign0][0]), int(points[sign0][1])),
                     (int(points[sign1][0]), int(points[sign1][1])),
                     (255, 255, 0), 3, cv2.LINE_AA)
        elif k <= 7 and k >= 4:
            cv2.line(image, (int(points[sign0][0]), int(points[sign0][1])),
                     (int(points[sign1][0]), int(points[sign1][1])),
                     (200, 158, 95), 3, cv2.LINE_AA)
        elif k <= 11 and k > 7:
            cv2.line(image, (int(points[sign0][0]), int(points[sign0][1])),
                     (int(points[sign1][0]), int(points[sign1][1])),
                     (215, 235, 250), 3, cv2.LINE_AA)
        else:
            cv2.line(image, (int(points[sign0][0]), int(points[sign0][1])),
                     (int(points[sign1][0]), int(points[sign1][1])),
                     (135, 184, 222), 3, cv2.LINE_AA)
        # cv2.line(image, (int(points[sign0 * 3]), int(points[sign0 * 3 + 1])),
        #          (int(points[sign1 * 3]), int(points[sign1 * 3 + 1])),
        #          (250, 235, 215), 1, cv2.LINE_AA)

    for j in range(17):
        # if keypoints[j * 3 + 2] < 0.05:
        #     continue
        center = tuple(map(int, points[j]))

        if vis_kpt[0][j]:
            cv2.circle(image, center=center, radius=1, color=(80, 127, 255), thickness=2)
            # plt.annotate(scores[j], center)
    return image


def draw_mmpose(img,
                keypoints,
                scores,
                keypoint_info,
                skeleton_info,
                kpt_thr=0.5,
                radius=2,
                line_width=2):
    assert len(keypoints.shape) == 2

    vis_kpt = [s >= kpt_thr for s in scores]

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'])
        link_dict[kpt_info['name']] = kpt_info['id']

        kpt = keypoints[i]

        if vis_kpt[i]:
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius),
                             kpt_color, -1)

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        if vis_kpt[pt0] and vis_kpt[pt1]:
            link_color = ske_info['color']
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]

            img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])),
                           (int(kpt1[0]), int(kpt1[1])),
                           link_color,
                           thickness=line_width)

    return img


def draw_openpose(img,
                  keypoints,
                  scores,
                  keypoint_info,
                  skeleton_info,
                  kpt_thr=0.4,
                  radius=4,
                  alpha=1.0,
                  line_width=2):
    h, w = img.shape[:2]

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'])
        link_dict[kpt_info['name']] = kpt_info['id']

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        link_color = ske_info['color']
        kpt0, kpt1 = keypoints[pt0], keypoints[pt1]
        s0, s1 = scores[pt0], scores[pt1]

        if (kpt0[0] <= 0 or kpt0[1] >= w or kpt0[1] <= 0 or kpt0[1] >= h
                or kpt1[0] <= 0 or kpt1[1] >= w or kpt1[1] <= 0 or kpt1[1] >= h
                or s0 < kpt_thr or s1 < kpt_thr or link_color is None):
            continue

        X = np.array([kpt0[0], kpt1[0]])
        Y = np.array([kpt0[1], kpt1[1]])

        if i <= 16:
            # body part
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            transparency = 0.6
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygons = cv2.ellipse2Poly((int(mX), int(mY)),
                                        (int(length / 2), int(line_width)),
                                        int(angle), 0, 360, 1)
            img = draw_polygons(img,
                                polygons,
                                edge_colors=link_color,
                                alpha=transparency)
        else:
            img = cv2.line(img, (int(X[0]), int(Y[0])), (int(X[1]), int(Y[1])),
                           link_color,
                           thickness=2)

    for j, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'][::-1])
        kpt = keypoints[j]

        if scores[j] < kpt_thr or sum(kpt_color) == 0:
            continue

        transparency = alpha
        if 24 <= j <= 91:
            j_radius = 3
        else:
            j_radius = 4
        # j_radius = radius // 2 if j > 17 else radius

        img = draw_circles(img,
                           kpt,
                           radius=np.array([j_radius]),
                           face_colors=kpt_color,
                           alpha=transparency)

    return img


def draw_polygons(img, polygons, edge_colors, alpha=1.0):
    if alpha == 1.0:
        img = cv2.fillConvexPoly(img, polygons, edge_colors)
    else:
        img = cv2.fillConvexPoly(img.copy(), polygons, edge_colors)
        img = cv2.addWeighted(img, 1 - alpha, img, alpha, 0)
    return img


def draw_circles(img, center, radius, face_colors, alpha=1.0):
    if alpha == 1.0:
        img = cv2.circle(img, (int(center[0]), int(center[1])), int(radius),
                         face_colors, -1)
    else:
        img = cv2.circle(img.copy(), (int(center[0]), int(center[1])),
                         int(radius), face_colors, -1)
        img = cv2.addWeighted(img, 1 - alpha, img, alpha, 0)
    return img
