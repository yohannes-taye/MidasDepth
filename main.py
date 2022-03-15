import cv2
import torch
import time
import numpy as np
import argparse
import tqdm 
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm 

parser = argparse.ArgumentParser(description="Freeze your network")
parser.add_argument("--img", type=str, help="Path to inference image folder", required=True)
parser.add_argument("--out", type=str, help="Path to output image folder", required=True)
parser.add_argument("--model", type=int, help="Which model to run (diffrent inference speed and quality of output) [1: High, 2: Mid, 3: Low]", required=True)
parser.add_argument("--debug", type=int, help="Display window and dont save output", default=0)

args = parser.parse_args()
opts = args 

if os.path.isfile(opts.img):
    img_list = [opts.img]
elif os.path.isdir(opts.img):
    img_list = glob.glob(os.path.join(opts.img, "*.{}".format("jpg")))
    img_list = sorted(img_list)
    if len(img_list) == 0:
        raise ValueError("No {} images found in folder {}".format(".png", opts.img))
    print("=> found {} images".format(len(img_list)))
else:
    raise Exception("No image nor folder provided")





# Load a MiDas model for depth estimation
if(args.model == 1): 
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
elif(args.model == 2): 
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
elif(args.model == 3): 
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda")# if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform




# Open up the video capture from a webcam
# cap = cv2.VideoCapture(2)
for i in tqdm(range(len(img_list))):
# while cap.isOpened():
    # success, img = cap.read()

    start = time.time()
    img = cv2.imread(img_list[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    # cv2.imshow('Image', numpy_horizontal)
    if(args.debug == 1): 
        cv2.imshow('Img', img)
        cv2.imshow('Depth', depth_map)
    else: 
        numpy_horizontal = np.hstack((img, depth_map))
        file_name = (img_list[i].split("/"))[-1]
        dest = f"{args.out}/{file_name}"
        cv2.imwrite(dest, numpy_horizontal)


    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()