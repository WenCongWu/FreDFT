from models.yolo_test import Model
import argparse
import torch
from utils.torch_utils import select_device
from utils.general import check_file, set_logging
from thop.profile import profile
from thop import clever_format
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./models/transformer/yolov5l_FreFT_M3FD.yaml', help='model.yaml')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)
    print(device)

    model = Model(opt.cfg).to(device)
    input_rgb = torch.Tensor(1, 3, 640, 640).to(device)
    input_ir = torch.Tensor(1, 3, 640, 640).to(device)

    #output = model(input_rgb, input_ir)

    macs, params = profile(model, inputs=(input_rgb, input_ir))
    macs, params = clever_format([2*macs, params], '%.4f')
    print('macs: ', macs)
    print('params: ', params)
