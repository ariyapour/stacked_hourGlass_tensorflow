import sys

import os
import numpy as np
import scipy.misc
from heatmap_process import post_process_heatmap
from keras.models import load_model, model_from_json
from hourGlass_CNN.hourGlass import stack_hourGlass_modules
import argparse
from pckh import run_pckh
from data_gen.mpii_datagen import MPIIDataGen
from data_gen.data_process import normalize
import cv2
import imageio
import time

def render_joints(cvmat, joints, conf_th=0.2):
    for _joint in joints:
        _x, _y, _conf = _joint
        if _conf > conf_th:
            cv2.circle(cvmat, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=2)

    return cvmat


def main_inference(model_json, model_weights, num_stack, num_class, imgfile, confth, tiny):

    with open(model_json) as f:
            model = model_from_json(f.read())
    model.load_weights(model_weights)

    # out, scale = xnet.inference_file(imgfile)
    #Inference
    image = imageio.imread(imgfile)
    scale = (image.shape[0] * 1.0 / 256, image.shape[1] * 1.0 / 256)
    imgdata = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
    mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
    imgdata = normalize(imgdata, mean)
    input = imgdata[np.newaxis, :, :, :]
    #inference time measurement
    start = time.time()
    out = model.predict(input)
    end = time.time()
    print("[INFO..] Inference time: {}".format(end-start))
    out = out[-1]
    kps = post_process_heatmap(out[0, :, :, :])

    ignore_kps = ['plevis', 'thorax', 'head_top']
    kp_keys = MPIIDataGen.get_kp_keys()
    mkps = list()
    for i, _kp in enumerate(kps):
        if kp_keys[i] in ignore_kps:
            _conf = 0.0
        else:
            _conf = _kp[2]
        mkps.append((_kp[0] * scale[1] * 4, _kp[1] * scale[0] * 4, _conf))

    cvmat = render_joints(cv2.imread(imgfile), mkps, confth)

    cv2.imshow('frame', cvmat)
    cv2.waitKey()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--model_json", help='path to store trained model')
    parser.add_argument("--model_weights", help='path to store trained model')
    parser.add_argument("--num_stack", type=int, help='num of stack')
    parser.add_argument("--input_image", help='input image file')
    parser.add_argument("--conf_threshold", type=float, default=0.2, help='confidence threshold')
    parser.add_argument("--tiny", default=False, type=bool, help="tiny network for speed, inres=[192x128], channel=128")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    main_inference(model_json=args.model_json, model_weights=args.model_weights, num_stack=args.num_stack,
                   num_class=16, imgfile=args.input_image, confth=args.conf_threshold, tiny=args.tiny)