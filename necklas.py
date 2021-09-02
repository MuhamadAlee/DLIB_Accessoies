# Facial landmarks with dlib, OpenCV, and PythonPython

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", default="model/shape_predictor_68_face_landmarks.dat", help="path to facial landmark predictor")
    ap.add_argument("-i", "--image", default="imgs/4.jpg", help="path to input image")
    ap.add_argument("-n", "--necklas", default="necklases/2.png", help="path to necklas image")
    args = vars(ap.parse_args())
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    image = cv2.imread(args["image"])
    nkl=cv2.imread(args["necklas"],-1)
    show_raw_detection(image,nkl, detector, predictor)


def show_raw_detection(image, nk_img, detector, predictor):
    b,g,r,a=cv2.split(nk_img)
    nk_img = cv2.merge((b,g,r))
    org_height, org_width, channel = nk_img.shape
    mask = cv2.medianBlur(a,5)
    mask_inv = cv2.bitwise_not(mask)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        ################ Calculating height and width ################
        wu=-1
        wl=-1
        hu=-1
        hl=-1

        for i,(x, y) in enumerate(shape):
            if i==0:
                wu=x
            elif i==16:
                wl=x

        w=(wl-wu)
        h=w//2

        #############################################################
        ####################### Calculate x,y########################

        X=-1
        Y=-1
        for i,(x, y) in enumerate(shape):
            if i==8:
                X=x
                Y=y+h

        #############################################################

        x1 = int(X - (w/2))
        x2 = int(X + (w/2))
        y1 = int(Y - (w/2))
        y2 = int(Y + (w/2))
        
        ht, wd = image.shape[:2]
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > wd:
            x2 = wd
        if y2 > ht:
            y2 = ht

        nk_width = int(x2 - x1)
        nk_height = int(y2 - y1)

        nkls = cv2.resize(nk_img, (nk_width,nk_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask, (nk_width,nk_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(mask_inv, (nk_width,nk_height), interpolation = cv2.INTER_AREA)

        roi = image[y1:y2, x1:x2]
        
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        
        # roi_fg contains the orignal location of earring
        roi_fg = cv2.bitwise_and(nkls,nkls,mask = mask)
        
        # joining the roi_bg and roi_fg
        dst = cv2.add(roi_bg,roi_fg)
        # placing the joined image and saving to dst back over the original image
        image[y1:y2, x1:x2] = dst


        
    
    
    cv2.imwrite("jhjd.jpg",image)
    cv2.imshow("Output", image)
    cv2.waitKey(0)




if __name__ == '__main__':
    main()

