import gradio as gr
import numpy as np

from rtmlib import Body, Wholebody, draw_skeleton

cached_model = {}


def predict(img,
            method1,
            method2,
            model_type,
            black_bg=False,
            backend='onnxruntime',
            device='cpu'):
    if model_type == 'body':
        constructor = Body
    elif model_type == 'wholebody':
        constructor = Wholebody
    else:
        raise NotImplementedError
    openpose_skeleton = False

    if method1 == 'HRNet':
        mode = 'balanced'
        det_input_size = (640, 640)
        pose_input_size = (288, 384)
        model_id_1 = str((constructor.__qualname__, det_input_size, pose_input_size, mode, openpose_skeleton, black_bg,
                          backend, device))
    elif method1 == 'DAG':
        mode = 'performance'
        det_input_size = (640, 640)
        pose_input_size = (192, 256)
        model_id_1 = str((constructor.__qualname__, det_input_size, pose_input_size, mode, openpose_skeleton, black_bg,
                          backend, device))
    elif method1 == 'MSS':
        mode = 'performance'
        det_input_size = (640, 640)
        pose_input_size = (192, 256)
        model_id_1 = str((constructor.__qualname__, det_input_size, pose_input_size, mode, openpose_skeleton, black_bg,
                          backend, device))
    elif method1 == 'RTMPose':
        mode = 'lightweight'
        det_input_size = (416, 416)
        pose_input_size = (192, 256)
        model_id_1 = str((constructor.__qualname__, det_input_size, pose_input_size, mode, openpose_skeleton, black_bg,
                          backend, device))

    if model_id_1 in cached_model:
        model_1 = cached_model[model_id_1]
    else:
        model_1 = constructor(
            det_input_size=det_input_size,
            pose_input_size=pose_input_size,
            mode=mode,
            to_openpose=openpose_skeleton,
            backend=backend,
            device=device)
        cached_model[model_id_1] = model_1

    if method2 == 'HRNet':
        mode = 'balanced'
        det_input_size = (640, 640)
        pose_input_size = (288, 384)
        model_id_2 = str((constructor.__qualname__, det_input_size, pose_input_size, mode, openpose_skeleton, black_bg,
                          backend, device))
    elif method2 == 'DAG':
        mode = 'performance'
        det_input_size = (640, 640)
        pose_input_size = (192, 256)
        model_id_2 = str((constructor.__qualname__, det_input_size, pose_input_size, mode, openpose_skeleton, black_bg,
                          backend, device))
    elif method2 == 'MSS':
        mode = 'performance'
        det_input_size = (640, 640)
        pose_input_size = (192, 256)
        model_id_2 = str((constructor.__qualname__, det_input_size, pose_input_size, mode, openpose_skeleton, black_bg,
                          backend, device))
    elif method2 == 'RTMPose':
        mode = 'lightweight'
        det_input_size = (416, 416)
        pose_input_size = (192, 256)
        model_id_2 = str((constructor.__qualname__, det_input_size, pose_input_size, mode, openpose_skeleton, black_bg,
                          backend, device))

    if model_id_2 in cached_model:
        model_2 = cached_model[model_id_2]
    else:
        model_2 = constructor(
            det_input_size=det_input_size,
            pose_input_size=pose_input_size,
            mode=mode,
            to_openpose=openpose_skeleton,
            backend=backend,
            device=device)

        cached_model[model_id_2] = model_2

    keypoints_1, scores_1 = model_1(img)
    keypoints_2, scores_2 = model_2(img)

    if black_bg:
        img_show = np.zeros_like(img, dtype=np.uint8)
    else:
        img_show = img.copy()

    img_show_1 = draw_skeleton(img_show,
                               keypoints_1,
                               scores_1,
                               openpose_skeleton=openpose_skeleton,
                               kpt_thr=0.4)

    img_show_2 = draw_skeleton(img_show,
                               keypoints_2,
                               scores_1,
                               openpose_skeleton=openpose_skeleton,
                               kpt_thr=0.4)
    return img_show_1[:, :, ::], img_show_2[:, :, ::]


with gr.Blocks(title='遮挡人体姿态估计算法系统') as demo:
    gr.Markdown('# 遮挡人体姿态估计算法系统')
    with gr.Tab('Upload-Image'):
        input_img = gr.Image(type='numpy')
        button = gr.Button('Inference', variant='primary')
        method1 = gr.Dropdown(['HRNet', 'DAG', 'MSS', 'RTMPose'],
                              label='算法1',
                              value='HRNet'
                              )
        method2 = gr.Dropdown(['HRNet', 'DAG', 'MSS', 'RTMPose'],
                              label='算法2',
                              value='DAG')
        model_type = gr.Dropdown(['body', 'wholebody'],
                                 label='关键点种类',
                                 info='Body / Wholebody',
                                 value='body')
        backend = gr.Dropdown(['opencv', 'onnxruntime'],
                              label='选择后台',
                              info='opencv / onnxruntime',
                              value='onnxruntime')
        device = gr.Dropdown(['cpu', 'cuda'],
                             label='选择推理设备',
                             info='cpu / cuda',
                             value='cpu')

        gr.Markdown('## Output')
        black_bg = gr.Checkbox(
            label='黑色背景',
            info='是否以黑色背景呈现骨架')
        with gr.Row():
            out_image_1 = gr.Image(type='numpy')
            out_image_2 = gr.Image(type='numpy')
        # gr.Examples(['./demo.jpg'], input_img)
        button.click(predict, inputs=[
            input_img, method1, method2, model_type, black_bg, backend, device
        ], outputs=[out_image_1, out_image_2])

gr.close_all()
demo.queue()
demo.launch(debug=True)
