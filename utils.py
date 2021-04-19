from difflib import SequenceMatcher

import xml.etree.ElementTree as ET
import re
import numpy as np
import cv2
from PIL import Image
from skimage.filters import rank
from skimage.morphology import disk

def cleanup_text(string:str):
    pattern = re.compile('[\W_]+')
    string = pattern.sub('', string)
    return string.lower()

def read_content(xml_file: str):

	tree = ET.parse(xml_file)
	root = tree.getroot()

	list_with_all_boxes = []

	for boxes in root.iter('object'):

		filename = root.find('filename').text

		ymin, xmin, ymax, xmax = None, None, None, None

		ymin = int(boxes.find("bndbox/ymin").text)
		xmin = int(boxes.find("bndbox/xmin").text)
		ymax = int(boxes.find("bndbox/ymax").text)
		xmax = int(boxes.find("bndbox/xmax").text)

		label = boxes.find("name").text

		list_with_single_boxes = [xmin, ymin, xmax, ymax]
		list_with_all_boxes.append((list_with_single_boxes,label))

	return filename, list_with_all_boxes


def similar(a:str, b:str):
	return SequenceMatcher(None, a, b).ratio()


def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou



def cvt_to_gray(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return img

def bilateral_filter(img,d, sigmaColor, sigmaSpace):

    return cv2.bilateralFilter(img,d,sigmaColor,sigmaSpace)

def apply_CAHE(img,clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    
    return clahe.apply(img)

def resize_image(img, new_height):
    height, width = img.shape[0], img.shape[1]
    ratio = width/height
    new_width = int(new_height * ratio)

    return cv2.resize(img,(new_width,new_height) , interpolation=Image.ANTIALIAS)

def equalizeColor(img):

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output

def make_black_dominant_color(img):
    pixel_counts = np.bincount(img.reshape(-1))

    black_pixels, white_pixels = pixel_counts[0], pixel_counts[-1]

    if white_pixels > black_pixels:
        img = 255 - img

    return img


def sharpening(image):
    kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)

    return sharpened

def otsu(image):

    #check if image is grayscale
    assert (len(image.shape)==2 or  (len(image.shape)==3 and image.shape[-1] == 1))
    
    image_blur = cv2.GaussianBlur(image,(5,5),0)
    _,image_th = cv2.threshold(image_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return image_th

def local_otsu(image, radius=15):
    # radius = 15
    selem = disk(radius)

    local_otsu = rank.otsu(image, selem)

    return (image >= local_otsu) * 255

def kmeans(image, K=8):

    Z = image.reshape(-1)

    Z = Z.astype('float32')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    # cv2.imshow('res2',res2)
    # cv2.waitKey(0)

    return res2

def adaptive_threshold(img):
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51,10)

    return img

def make_white_dominant(img):
    black_pixels = np.sum(np.where(img[:,0]==255,1,0))
    if black_pixels > (img.shape[1]//2):
        img = 255-img

    return img

def equalize_hist(img):
    return cv2.equalizeHist(img)