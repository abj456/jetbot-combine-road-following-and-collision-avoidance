import base64
from io import BytesIO
import requests
from urllib.request import urlopen
from argparse import ArgumentParser
import sys
import json

import torchvision
import torch
from torch2trt import TRTModule, torch2trt
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
# import cv2
import numpy as np
import flask
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = flask.Flask(__name__)
data_transforms = None

###############################
'''load_model'''
def load_model(path_r, path_c):
    data = torch.zeros((1, 3, 224, 224)).cuda().half()
    # model_road = TRTModule()
    # model_road.load_state_dict(torch.load(path_r))
    model_road = torchvision.models.resnet18(pretrained=False)
    model_road.fc = torch.nn.Linear(512, 2)
    model_road.load_state_dict(torch.load(path_r))
    model_road = model_road.to(device)
    model_road = model_road.eval().half()
    model_road_trt = torch2trt(model_road, [data], fp16_mode=True)

    # model_coll = TRTModule()
    # model_coll.load_state_dict(torch.load(path_c))
    model_coll = torchvision.models.resnet18(pretrained=False)
    model_coll.fc = torch.nn.Linear(512, 3)
    model_coll.load_state_dict(torch.load(path_c))
    model_coll = model_coll.to(device)
    model_coll = model_coll.eval().half()
    model_coll_trt = torch2trt(model_coll, [data], fp16_mode=True)

    return (model_road_trt, model_coll_trt)

###############################
'''pre-processing function'''
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray((image * 255).astype(np.uint8))
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
#################################
@app.route("/predict", methods=["POST"])
def predict():
    # print(flask.request.json)

    output_dict = {"success": False}
    if flask.request.method == "POST":
        #im_b64 = flask.request.json['image']
        #img_bytes = base64.b64decode(im_b64.encode('utf-8'))
        #image = PIL.Image.open(BytesIO(img_bytes))
        image = flask.request.json['image']
        img_arr = np.asarray(image)
        # print('img shape', img_arr.shape)
        
        # predict image - road following
        xy_r = model_road(preprocess(img_arr)).detach().float().cpu().numpy().flatten()
        x_r = xy_r[0]
        y_r = (0.5 - xy_r[1]) / 2.0

        # predict image - collision avoidance
        x_c = model_coll(preprocess(img_arr))
        x_c = F.softmax(x_c, dim=1)

        prob_blocked_l = float(x_c.flatten()[0])
        prob_blocked_r = float(x_c.flatten()[1])

        # end
        output_dict["road following x_r"] = np.float(x_r)
        output_dict["road following y_r"] = np.float(y_r)
        output_dict["collision avoidance left"] = np.float(prob_blocked_l)
        output_dict["collision avoidance right"] = np.float(prob_blocked_r)
        output_dict["success"] = True
        # print(output_dict)
    return flask.jsonify(output_dict), 200
#################################
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_r', help='model_road path', type=str, default=None)
    parser.add_argument('--model_c', help='model_coll path', type=str, default=None)
    parser.add_argument('--port', help='port', type=int, default=5000)
    args = parser.parse_args()
    print(("* Loading pytorch model and Flask starting server... please wait until server has fully started"))
    if (args.model_r is None):
        print("You have to load the model_r by --model_r")
        sys.exit()
    if (args.model_c is None):
        print("You have to load the model_c by --model_c")
        sys.exit()
    # print(args.model_r, args.model_c)
    model_road, model_coll = load_model(args.model_r, args.model_c)
    app.run(host="0.0.0.0", debug=True, port=args.port)

