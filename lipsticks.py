
import cv2,sys,dlib,time,math
from imutils import face_utils
import numpy as np
import os
import colorsys

PREDICTOR_PATH =  "./model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

image = cv2.imread("imgs/5.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)
points=[]
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)    
    for s in shape:
        points.append(tuple(s))
    break

upperlips = points[48:55] + points[60:65][::-1]
lowerlips = points[48:49] + points[54:60][::-1] + points[64:]


uHull = [[p[0],p[1]] for p in upperlips]
lHull = [[p[0],p[1]] for p in lowerlips]

uHull = np.array(uHull)
lHull = np.array(lHull)

row, col, _ = image.shape
mask = np.zeros((row, col), dtype=image.dtype)
cv2.fillPoly(mask, [uHull], (255,255,255))
cv2.fillPoly(mask, [lHull], (255,255,255))
bit_mask = mask.astype(np.bool)


lst = upperlips + lowerlips
xmin, xmax = min(lst, key = lambda i : i[1])[1], max(lst, key = lambda i : i[1])[1]
ymin, ymax = min(lst, key = lambda i : i[0])[0], max(lst, key = lambda i : i[0])[0]

color = (175, 125, 75)
pixel = np.zeros((1,1,3), dtype=np.uint8)
r_ = 0
g_ = 1
b_ = 2

pixel[:,:,r_], pixel[:,:,g_], pixel[:,:,b_] = color[r_], color[g_], color[b_]



out = image.copy()

# Convert image of person from RGB to HLS
pixel_hsl = cv2.cvtColor(pixel, cv2.COLOR_RGB2HLS)
outhsv = cv2.cvtColor(out,cv2.COLOR_RGB2HLS)
channel = 0

# extract the hue channel
hue_img = outhsv[:,:,channel]
hue_pixel = pixel_hsl[:,:,0]

hue_img[bit_mask] = hue_pixel[0,0]

out = cv2.cvtColor(outhsv,cv2.COLOR_HLS2RGB)


cv2.imwrite("res.png",out)
