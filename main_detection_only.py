import easyocr
from easyocr import Reader
import argparse
import cv2
import os
import torch
import numpy as np
from skimage.filters import rank , threshold_otsu
from skimage.morphology import disk
import io
import json
from PIL import Image
from sklearn.metrics import average_precision_score

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

# def cvt_to_gray(img):
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     return img

# def bilateral_filter(img,d, sigmaColor, sigmaSpace):

#     return cv2.bilateralFilter(img,d,sigmaColor,sigmaSpace)

# def apply_CAHE(img,clipLimit=2.0, tileGridSize=(8,8)):
#     clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    
#     return clahe.apply(img)

# def resize_image(img, new_height):
#     height, width = img.shape[0], img.shape[1]
#     ratio = width/height
#     new_width = int(new_height * ratio)

#     return cv2.resize(img_cropped,(new_width,new_height) , interpolation=Image.ANTIALIAS)

# def equalizeColor(img):

#     img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#     img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#     img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

#     return img_output

# def make_black_dominant_color(img):
#     pixel_counts = np.bincount(img.reshape(-1))

#     black_pixels, white_pixels = pixel_counts[0], pixel_counts[-1]

#     if white_pixels > black_pixels:
#         img = 255 - img

#     return img

# def read_content(xml_file: str):

#     tree = ET.parse(xml_file)
#     root = tree.getroot()

#     list_with_all_boxes = []

#     for boxes in root.iter('object'):

#         filename = root.find('filename').text

#         ymin, xmin, ymax, xmax = None, None, None, None

#         ymin = int(boxes.find("bndbox/ymin").text)
#         xmin = int(boxes.find("bndbox/xmin").text)
#         ymax = int(boxes.find("bndbox/ymax").text)
#         xmax = int(boxes.find("bndbox/xmax").text)

#         label = boxes.find("name").text

#         list_with_single_boxes = [xmin, ymin, xmax, ymax]
#         list_with_all_boxes.append((list_with_single_boxes,label))

#     return filename, list_with_all_boxes



# def similar(a:str, b:str):
#     return SequenceMatcher(None, a, b).ratio()


# def bb_iou(boxA, boxB):
#     # determine the (x, y)-coordinates of the intersection rectangle
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     # compute the area of intersection rectangle
#     interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
#     if interArea == 0:
#         return 0
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
#     boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = interArea / float(boxAArea + boxBArea - interArea)

#     # return the intersection over union value
#     return iou

# def cleanup_text(text):
#     # strip out non-ASCII text so we can draw the text on the image
#     # using OpenCV
#     # return "".join([c if ord(c) < 128 else "" for c in text]).strip()

#     return text.replace(" ","")



# def sharpening(image):
#     kernel_sharpening = np.array([[-1,-1,-1], 
#                               [-1, 9,-1],
#                               [-1,-1,-1]])
#     # applying the sharpening kernel to the input image & displaying it.
#     sharpened = cv2.filter2D(image, -1, kernel_sharpening)
#     # cv2.imshow('Image Sharpening', sharpened)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     return sharpened

# def otsu(image):

#     #check if image is grayscale
#     assert (len(image.shape)==2 or  (len(image.shape)==3 and image.shape[-1] == 1))
    
#     image_blur = cv2.GaussianBlur(image,(5,5),0)
#     _,image_th = cv2.threshold(image_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#     return image_th

# def local_otsu(image, radius=15):
#     # radius = 15
#     selem = disk(radius)

#     local_otsu = rank.otsu(image, selem)

#     return (image >= local_otsu) * 255

# def kmeans(image, K=8):

#     Z = image.reshape(-1)

#     Z = Z.astype('float32')

#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#     # Now convert back into uint8, and make original image
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     res2 = res.reshape((image.shape))

#     # cv2.imshow('res2',res2)
#     # cv2.waitKey(0)

#     return res2


# def remove_noise_and_smooth(img):
#     filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
#     kernel = np.ones((1, 1), np.uint8)
#     opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

#     # cv2.imshow("Image", closing)
#     # cv2.waitKey(0)

#     blur = cv2.GaussianBlur(img,(5,5),0)

#     # cv2.imshow("Image", blur)
#     # cv2.waitKey(0)
#     or_image = cv2.bitwise_and(blur, closing)
#     return or_image


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
print("[INFO] OCR'ing with the following languages: {}".format(langs))
reader = Reader(langs,gpu=True)

# load the input image from disk

# detection_pipeline= [{"function":"cvt_to_gray", "kwargs":{}}, {"function":"apply_CAHE", "kwargs":{"clipLimit":2.0, "tileGridSize":(8,8)}}]
# detection_pipeline= [{"function":"cvt_to_gray", "kwargs":{}}]
detection_pipeline = []
# detection_pipeline= [{"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]
# detection_pipeline= [{"function":"cvt_to_gray", "kwargs":{}},{"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]
# detection_pipeline= [{"function":"sharpening", "kwargs":{}}]
# detection_pipeline= [{"function":"sharpening", "kwargs":{}},{"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]
# detection_pipeline= [{"function":"cvt_to_gray", "kwargs":{}},{"function":"sharpening", "kwargs":{}},{"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]
# detection_pipeline= [{"function":"cvt_to_gray", "kwargs":{}}, {"function":"apply_CAHE", "kwargs":{"clipLimit":2.0, "tileGridSize":(8,8)}},{"function":"bilateral_filter", "kwargs":{"d":5, "sigmaColor":50, "sigmaSpace":50}}]
# detection_pipeline= [{"function":"cvt_to_gray", "kwargs":{}},{"function":"otsu", "kwargs":{}}]
# recognition_pipeline = [{"function":"cvt_to_gray", "kwargs":{}},{"function":"resize_image", "kwargs":{"new_height":128}}, {"function":"bilateral_filter", "kwargs":{"d":5, "sigmaColor":50, "sigmaSpace":150}}]

results_path= "results_new/results_"+now.strftime("%Y_%m_%d_%H_%M_%S")+"/"

os.makedirs(results_path)

false_positives = 0
true_positives = 0
false_negatives = 0
correct_classified = 0

sum_similarity = 0
resultsText = ""
pbar =  tqdm(os.listdir(args["Images"]))

y_true = []
y_pred = []

box_results=[]

gt_count = 0

for file in pbar:

    if not file.endswith(".jpg"):
        continue

    

    image = cv2.imread(os.path.join(args["Images"],file))

    try:
        _ , boxes = read_content(os.path.join(args["Images"],file.split(".")[0]+".xml"))
    except:
        continue


    gt_count += len(boxes)
    classifications_per_box= [0 for _ in range(len(boxes))]

    image_detection = np.copy(image)

    for step in detection_pipeline:
        func = step["function"]
        func = globals()[func]
        kwargs = step["kwargs"]

        image_detection = func(image_detection, **kwargs)

    
    image_detection = image_detection.astype('uint8')

    boxes_detected = reader.detect(image_detection)

    image_res = np.copy(image_detection)


    resultsText+=file +"\n"

    results_json= {"filename":file,"boxes":[]}

    fps = 0
    tps = 0
    fns = 0
    ccs = 0

    empty = np.zeros_like(image_res)

    IoU_threshold = 0.5


    if not os.path.exists(results_path+file.split(".")[0]):
        os.makedirs(results_path+file.split(".")[0])

    for i,box in enumerate(boxes_detected[0]):

        min_x, max_x, min_y , max_y = box

        width , height = max_x-min_x, max_y-min_y

        min_x ,max_x,min_y,max_y = max(0,min_x), min(image.shape[1]-1,max_x), max(0,min_y), min(image.shape[0]-1,max_y)

        img_cropped = image[ min_y:max_y,min_x:max_x]

        img_cropped = img_cropped.astype('uint8')

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


        
        resultsText+= f'Detected Box ({min_x,min_y,max_x,max_y}) '
        
        # text = utils.cleanup_text("")
        # box_type= "FN"
        if maxIoU < IoU_threshold:
            false_positives+=1
            fps+=1
            resultsText+="False positive\n"
            box_type="FP"
            y_true.append(0)
            y_pred.append(maxIoU)
        else:
            true_positives+=1
            tps+=1
            classifications_per_box[idxGT]+=1
            textGT = utils.cleanup_text(labelGT)
            box_type="TP"
            resultsText+="True positive True text: "+ labelGT

            resultsText+="\n"
            y_true.append(1)
            y_pred.append(maxIoU)



        results_json["boxes"].append({"min_x":int(min_x), "min_y": int(min_y), "max_x":int(max_x), "max_y":int(max_y), "text":text})

        results_json["boxes"][i]["type"]=box_type

        
        if boxGT is not None:
            cv2.rectangle(image_res, (boxGT[0],boxGT[1]), (boxGT[2],boxGT[3]), (0, 0, 255), 2)
            # cv2.putText(image_res, labelGT, (boxGT[0], boxGT[1] - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.rectangle(image_res, (min_x,min_y),(max_x,max_y), (0,255,0), 2)
        # cv2.putText(image_res, text, (max_x-20, min_y - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255,0), 2)

        del img_cropped

    pbar.set_description(f'Precision: {true_positives/(true_positives+false_positives):2.2%}  Recall: {true_positives/(true_positives+false_negatives):2.2%}  Accuracy: {correct_classified/true_positives:2.2%} Similarity:{sum_similarity/true_positives:2.2%}')

    for j in classifications_per_box:
        if j==0:
            false_negatives+=1
            fns+=1
            y_true.append(1)
            y_pred.append(0)

    

    results_json["boxesCount"]=len(boxes)
    results_json["fps"]=int(fps)
    results_json["tps"]=int(tps)
    results_json["fns"]=int(fns)
    results_json["ccs"]=int(ccs)


    with open(results_path+file.split(".")[0]+"/results.json","w") as f:
        json.dump(results_json,f)#
    cv2.imwrite(results_path+file.split(".")[0]+"/result.jpg",image_res)
    cv2.imwrite(results_path+file.split(".")[0]+"/result_ai.jpg",empty)

    del image
    torch.cuda.empty_cache()

with io.open("results/results.txt", "w",encoding='utf8') as f:
    f.write(json.dumps(detection_pipeline))
    f.write(resultsText)
    f.write(f'Precision:{true_positives/(true_positives+false_positives)}\nRecall:{true_positives/(true_positives+false_negatives)}\nAccuracy:{correct_classified/true_positives}')
    f.write(f'\nyAverage Precision:{average_precision_score(y_true,y_pred)}')
    f.write(json.dumps({"y_true": y_true, "y_pred":y_pred}))

with open(results_path+"/results.txt", "w") as f:
    f.write(resultsText)
    f.write(f'Precision:{true_positives/(true_positives+false_positives)}\nRecall:{true_positives/(true_positives+false_negatives)}\nAccuracy:{correct_classified/true_positives}')

with open(results_path+"/results.json", "w") as f:
    json.dump({"detection_pipeline":detection_pipeline, "y_true": y_true, "y_pred":y_pred},f)


print(f'Precision: {true_positives/(true_positives+false_positives):2.2%}')
print(f'Recall: {true_positives/(true_positives+false_negatives):2.2%}')
