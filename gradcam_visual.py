import os
#os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import time
import argparse
import numpy as np
from models.gradcam import YOLOV5GradCAM
from models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
import cv2
from models.experimental import attempt_load
#from deep_utils import Box, split_extension
from utils.general import set_logging
from utils.torch_utils import select_device
import torch

#names = ['person', 'car', 'bicycle'] # class names

#target = ['model-4-cv3-act', 'model-14-cv3-act']
#target = ['model-6-cv3-act', 'model-16-cv3-act']
#target = ['model-9-cv2-act', 'model_19-cv2-act']
#target = ['model-23-concat']
#target = ['model-24-concat']
#target = ['model-25-concat']
#target = ['model-23-conv1x1_out-act']
#target = ['model-24-conv1x1_out-act']
#target = ['model-25-conv1x1_out-act']
#target  = ['model-33-cv3-act']
#target = ['model-36-cv3-act']
#target = ['model-39-cv3-act']
target = ['model-28-attn']


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="./weights/FreDFT_FLIR.pt", help='Path to the model')
    parser.add_argument('--source1', type=str, default='./dataset/FLIR-align-3class/visible/test', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source2', type=str, default='./dataset/FLIR-align-3class/infrared/test', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output-dir', type=str, default='./runs/Grad', help='output dir')
    parser.add_argument('--img_size', type=int, default=640, help="input image size")
    parser.add_argument('--target-layer', type=str, default=target, help='The layer hierarchical address to which gradcam will applied,'
                             ' the names should be separated by underline')
    parser.add_argument('--method', type=str, default='gradcam', help='gradcam or gradcampp')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--no_text_box', action='store_true', help='do not show label and box on the heatmap')
    parser.add_argument('--names', type=str, default='person,car,bicycle',
                        help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
    # 'person, car, bicycle'
    args = parser.parse_args()


def get_res_img2(heat, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (heatmap / 255).astype(np.float32)
    heat.append(n_heatmat)
    return res_img, heat


def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    #n_heatmat = (Box.fill_outer_box(heatmap, bbox, value=0) / 255).astype(np.float32)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = cv2.addWeighted(res_img, 0.7, n_heatmat, 0.3, 0)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat


def split_extension(path, extension=None, suffix=None):
    remain, extension_ = os.path.splitext(path)
    if extension and suffix:
        return remain + suffix + extension
    elif extension is None and suffix:
        return remain + suffix + extension_
    elif extension:
        return remain + extension
    return remain, extension_

def concat_images(images):
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        base_img[:, h * i:h * (i + 1), ...] = img
    return base_img


def main(img_vis_path, img_ir_path):
    device = args.device
    input_size = (args.img_size, args.img_size)
    img_vis, img_ir = cv2.imread(img_vis_path), cv2.imread(img_ir_path)
    print('[INFO] Loading the model')
    # load model
    model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","), confidence=0.3)
    print(model)
    # preprocess the images
    torch_img_vis, torch_img_ir = model.preprocessing(img_vis[..., ::-1], img_ir[..., ::-1])
    print(torch_img_vis.shape)
    result = torch_img_vis.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr
    images = []
    if args.method == 'gradcam':
        for layer in args.target_layer:
            saliency_method = YOLOV5GradCAM(model=model, layer_name=layer, img_size=input_size)
            tic = time.time()
            masks, logits, [boxes, _, class_names, confs] = saliency_method(torch_img_vis, torch_img_ir)
            print("total time:", round(time.time() - tic, 4))
            res_img = result.copy()
            res_img = res_img / 255
            heat = []
            for i, mask in enumerate(masks):
                bbox = boxes[0][i]
                mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                n_heatmat = (heatmap / 255).astype(np.float32)
                heat.append(n_heatmat)
                #res_img, heat_map = get_res_img(bbox, mask, res_img)
                #res_img = put_text_box(bbox, cls_name, res_img)  # plot the bboxes
                #images.append(res_img)

            if(len(heat) != 0):
                heat_all = heat[0]
                for h in heat[1:]:
                    heat_all += h
                heat_avg = heat_all / len(heat)
                res_img = cv2.addWeighted(res_img, 0.3, heat_avg, 0.7, 0)
            res_img = (res_img / res_img.max())
            cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
            heat_map = cv2.imread('temp.jpg')
            # for i, mask in enumerate(masks):
            #     bbox, cls_name, conf = boxes[0][i], class_names[0][i], str(confs[0][i])
            #     heat_map = put_text_box(bbox, conf, heat_avg)  # plot the bboxes
            final_image = heat_map
            images.append(final_image)
            # save the images
            suffix = '-res-' + layer
            img_name = split_extension(os.path.split(img_vis_path)[-1], suffix=suffix)
            output_path = f'{args.output_dir}/{img_name}'
            os.makedirs(args.output_dir, exist_ok=True)
            print(f'[INFO] Saving the final image at {output_path}')
            cv2.imwrite(output_path, final_image)

        img_name = split_extension(os.path.split(img_vis_path)[-1], suffix='_avg')
        output_path = f'{args.output_dir}/{img_name}'
        img_all = images[0].astype(np.uint16)
        for img in images[1:]:
            img_all += img
        img_avg = img_all / len(images)
        cv2.imwrite(output_path, img_avg.astype(np.uint8))


if __name__ == '__main__':
    if os.path.isdir(args.source1):
        img_vis_list = os.listdir(args.source1)
        img_vis_list.sort()
        print(img_vis_list)

        for item in img_vis_list:
            print(item)
            img_vis_path = os.path.join(args.source1 ,item)
            if args.source1 == './dataset/FLIR_test/visible/test':
                new_item = item[:-4] + '.jpeg'
                img_ir_path = os.path.join(args.source2, new_item)
                print(img_ir_path)
            else:
                img_ir_path = os.path.join(args.source2, item)
            main(img_vis_path, img_ir_path)

    #else:
        #main(img_vis_path, img_ir_path)