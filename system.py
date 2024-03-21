import gradio as gr
import numpy as np
import cv2
from rtmlib import Body, Wholebody, draw_skeleton

cached_model = {}


def predict(img,
            openpose_skeleton=False,
            model_type='body',
            black_bg=False,
            backend='onnxruntime',
            device='cpu'):
    if model_type == 'body':
        constructor = Body
    elif model_type == 'wholebody':
        constructor = Wholebody
    else:
        raise NotImplementedError

    model_id = str((constructor.__qualname__, openpose_skeleton, black_bg,
                    backend, device))

    if model_id in cached_model:
        model = cached_model[model_id]
    else:
        model = constructor(to_openpose=openpose_skeleton,
                            backend=backend,
                            device=device)
        cached_model[model_id] = model

    keypoints, scores = model(img)

    if black_bg:
        img_show = np.zeros_like(img, dtype=np.uint8)
    else:
        img_show = img.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.4)
    # cv2.imshow('img', img_show)
    return img_show[:, :, ::]

    # return keypoints, scores


title = 'Human pose estimation'

# img = cv2.imread('./demo.jpg')

gr.Interface(inputs=['image'],
             outputs=['image'],
             fn=predict,
             title=title).launch(debug=True)
