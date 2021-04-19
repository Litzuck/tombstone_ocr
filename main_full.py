import easyocr
from easyocr import Reader
import argparse
import cv2
import os
import torch
import numpy as np
from skimage.filters import rank
from skimage.morphology import disk
import io
import json
from PIL import Image

from scipy import ndimage as ndi

import utils

from skimage.filters import threshold_multiotsu

import matplotlib.pyplot as plt

from tqdm import tqdm

from difflib import SequenceMatcher

import xml.etree.ElementTree as ET

import pytesseract
from pytesseract import Output

from datetime import datetime

from utils import *


pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
custom_oem_psm_config = r'--oem 1 --psm 13'

ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image to be OCR'd")
ap.add_argument("-I", "--Images", required=True)
ap.add_argument("-l", "--langs", type=str, default="en",
    help="comma separated list of languages to OCR")
ap.add_argument("-g", "--gpu", type=int, default=-1,
    help="whether or not GPU should be used")
args = vars(ap.parse_args())


now = datetime.now()
# break the input languages into a comma separated list
langs = args["langs"].split(",")
use_tesseract=True
print("[INFO] OCR'ing with the following languages: {}".format(langs))
reader = Reader(langs,gpu=True)

detection_pipeline= [{"function":"sharpening", "kwargs":{}},{"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]
recognition_pipeline = [{"function":"cvt_to_gray", "kwargs":{}},{"function":"resize_image", "kwargs":{"new_height":128}}, {"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]
recognition_pipeline = [{"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]


results_path= "results_full/results_"+now.strftime("%Y_%m_%d_%H_%M_%S")+"/"

os.makedirs(results_path)

false_positives = 0
true_positives = 0
false_negatives = 0
correct_classified = 0

sum_similarity = 0
resultsText = ""
pbar =  tqdm(os.listdir(args["Images"]))

for file in pbar:

    if not file.endswith(".jpg"):
        continue
    
    image = cv2.imread(os.path.join(args["Images"],file))

    try:
        _ , boxes = read_content(os.path.join(args["Images"],file.split(".")[0]+".xml"))
    except:
        continue


    classifications_per_box= [0 for _ in range(len(boxes))]

    image_detection = np.copy(image)

    ###Preprocessing of whole image
    for step in detection_pipeline:
        func = step["function"]
        func = globals()[func]
        kwargs = step["kwargs"]

        image_detection = func(image_detection, **kwargs)

    boxs = reader.detect(image_detection)

    image_res = np.copy(image_detection)

    resultsText+=file +"\n"

    results_json= {"filename":file,"boxes":[]}

    fps = 0
    tps = 0
    fns = 0
    ccs = 0

    empty = np.zeros_like(image_res)

    for i,box in enumerate(boxs[0]):

        min_x, max_x, min_y , max_y = box

        width , height = max_x-min_x, max_y-min_y

        min_x ,max_x,min_y,max_y = max(0,min_x), min(image.shape[1]-1,max_x), max(0,min_y), min(image.shape[0]-1,max_y)

        img_cropped = image[ min_y:max_y,min_x:max_x]

        ###Preprocessing for cropped image
        for step in recognition_pipeline:
            func = step["function"]
            func = globals()[func]
            kwargs = step["kwargs"]

            img_cropped = func(img_cropped, **kwargs)

        img_cropped = img_cropped.astype('uint8')

        if not os.path.exists(results_path+file.split(".")[0]):
            os.makedirs(results_path+file.split(".")[0])
        cv2.imwrite(results_path+file.split(".")[0]+"/"+str(i)+".jpg", img_cropped)


        maxIoU =0
        boxGT = None
        labelGT = None
        idxGT = None

        bbox = (min_x,min_y,max_x,max_y)
        for idx, (boxT, label) in enumerate(boxes):
            iou = bb_iou(bbox,boxT)

            if iou > maxIoU:
                maxIoU = iou
                boxGT = boxT
                labelGT = label
                idxGT = idx

        if use_tesseract:
            text =  pytesseract.image_to_string(img_cropped, lang="deu+frk",config=custom_oem_psm_config)
        else:
            result = reader.recognize(img_cropped)
            _,text,prob =result[0]

        
        
        resultsText+= f'Detected Box ({min_x,min_y,max_x,max_y}) '
        
        text = utils.cleanup_text(text)
        box_type= "FN"
        if maxIoU <0.5:
            false_positives+=1
            fps+=1
            resultsText+="False positive\n"
            box_type="FP"
        else:
            true_positives+=1
            tps+=1
            classifications_per_box[idxGT]+=1
            textGT = utils.cleanup_text(labelGT)
            box_type="TP"
            resultsText+="True positive True text: "+ labelGT+" Detected text: "+ text
            sum_similarity += similar(text,textGT)

            # if similar(text,labelGT)>0.85:
            # 	correct_classified+=1
            if text == textGT:
                correct_classified+=1
                ccs+=1
                resultsText+= " correctly identified"
                box_type="CC"
            resultsText+="\n"


        results_json["boxes"].append({"min_x":int(min_x), "min_y": int(min_y), "max_x":int(max_x), "max_y":int(max_y), "text":text})

        results_json["boxes"][i]["type"]=box_type

        
        if boxGT is not None:
            cv2.rectangle(image_res, (boxGT[0],boxGT[1]), (boxGT[2],boxGT[3]), (0, 0, 0), 5)
            cv2.putText(image_res, labelGT, (boxGT[0], boxGT[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 5)

        cv2.rectangle(image_res, (min_x,min_y),(max_x,max_y), (255,0,0), 5)
        cv2.putText(image_res, text, (max_x-20, min_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0,0), 5)

        del img_cropped

    pbar.set_description(f'Precision: {true_positives/(true_positives+false_positives):2.2%}  Recall: {true_positives/(true_positives+false_negatives):2.2%}  Accuracy: {correct_classified/true_positives:2.2%} Similarity:{sum_similarity/true_positives:2.2%}')

    for j in classifications_per_box:
        if j==0:
            false_negatives+=1
            fns+=1
            box_type="FN"

    results_json["boxesCount"]=len(boxes)
    results_json["fps"]=int(fps)
    results_json["tps"]=int(tps)
    results_json["fns"]=int(fns)
    results_json["ccs"]=int(ccs)


    with open(results_path+file.split(".")[0]+"/results.json","w")as f:
        json.dump(results_json,f)#
    cv2.imwrite(results_path+file.split(".")[0]+"/result.jpg",image_res)
    cv2.imwrite(results_path+file.split(".")[0]+"/result_ai.jpg",empty)

    del image
    torch.cuda.empty_cache()

with open(results_path+"/results.txt", "w",encoding='utf8') as f:
    f.write(resultsText)
    f.write(f'Precision:{true_positives/(true_positives+false_positives)}\nRecall:{true_positives/(true_positives+false_negatives)}\nAccuracy:{correct_classified/true_positives}')

with open(results_path+"/results.json", "w") as f:
    json.dump({"detection_pipeline":detection_pipeline, "recognition_pipeline":recognition_pipeline},f)


print(detection_pipeline)
print(recognition_pipeline)
print(true_positives/(true_positives+false_positives))
print(true_positives/(true_positives+false_negatives))
print(correct_classified/true_positives)
