
###DETECTION

#baseling
detection_pipeline = []
#grayscale
detection_pipeline= [{"function":"cvt_to_gray", "kwargs":{}}]
#grayscale and CAHE
detection_pipeline= [{"function":"cvt_to_gray", "kwargs":{}}, {"function":"apply_CAHE", "kwargs":{"clipLimit":2.0, "tileGridSize":(8,8)}}]
#bilateral filter
detection_pipeline= [{"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]
#grayscale, CAHE and bilateral filter
detection_pipeline= [{"function":"cvt_to_gray", "kwargs":{}}, {"function":"apply_CAHE", "kwargs":{"clipLimit":2.0, "tileGridSize":(8,8)}},{"function":"bilateral_filter", "kwargs":{"d":5, "sigmaColor":50, "sigmaSpace":50}}]
#sharpening mask
detection_pipeline= [{"function":"sharpening", "kwargs":{}}]
#sharpening mask and bilateral filter
detection_pipeline= [{"function":"sharpening", "kwargs":{}},{"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]



###RECOGNITION
#Baseline
recognition_pipeline = []
#grayscale
recognition_pipeline = [{"function":"cvt_to_gray", "kwargs":{}}]

#grayscale and CAHE
recognition_pipeline= [{"function":"cvt_to_gray", "kwargs":{}}, {"function":"apply_CAHE", "kwargs":{"clipLimit":2.0, "tileGridSize":(8,8)}}]
#bilateral filter
recognition_pipeline = [{"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]
#grayscale and bilateral filter
recognition_pipeline = [{"function":"cvt_to_gray", "kwargs":{}},{"function":"bilateral_filter", "kwargs":{"d":7, "sigmaColor":50, "sigmaSpace":50}}]
#grayscale and otsu
recognition_pipeline = [{"function":"cvt_to_gray", "kwargs":{}},{"function":"otsu", "kwargs":{}}]
#grayscale and local otsu
recognition_pipeline = [{"function":"cvt_to_gray", "kwargs":{}},{"function":"local_otsu", "kwargs":{}}]
#sharpening mask
recognition_pipeline = [{"function":"sharpening", "kwargs":{}}]
#adaptive thresholding
recognition_pipeline = [{"function":"cvt_to_gray", "kwargs":{}},{"function":"adaptive_threshold", "kwargs":{}}]
