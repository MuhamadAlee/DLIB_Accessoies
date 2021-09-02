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
    ap.add_argument("-i", "--image", default="imgs/2.jpg", help="path to input image")
    ap.add_argument("-e", "--earring", default="earrings/2.png", help="path to earRings image")
    args = vars(ap.parse_args())
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    image = cv2.imread(args["image"])
    ear=cv2.imread(args["earring"],-1)
    show_raw_detection(image,ear, detector, predictor)


def show_raw_detection(image, ear_img, detector, predictor):

    b,g,r,a=cv2.split(ear_img)
    ear_img = cv2.merge((b,g,r))
    org_height, org_width, channel = ear_img.shape
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
        hul=-1
        hll=-1
        hur=-1
        hlr=-1
        
        for i,(x, y) in enumerate(shape):
            if i==0:
                hul=y
            elif i==2:
                hll=y
            elif i==16:
                hur=y
            elif i==14:
                hlr=y
            elif i==36:
                wu=x
            elif i==39:
                wl=x


        w=(wl-wu)//3
        hl=(hll-hul)
        hr=(hlr-hur)

        Xl=-1
        Yl=-1
        Xr=-1
        Yr=-1
        for i,(x, y) in enumerate(shape):
            if i==0:
                Xl=x-w
                Yl=y

            if i==16:
                Xr=x
                Yr=y

    ###############################################################
    ########################Applying to Left Ear###################
    earring_height = hl
    earring_width = earring_height * org_width / org_height

    x1 = int(Xl - (earring_width/4))
    x2 = int(Xl + w + (earring_width/4))
    y1 = int(Yl + hl - (earring_height/2))
    y2 = int(Yl + hl + (earring_height/2))

    earring_width = int(x2 - x1)
    earring_height = int(y2 - y1)

    earring = cv2.resize(ear_img, (earring_width,earring_height), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (earring_width,earring_height), interpolation = cv2.INTER_AREA)
    mask_inv = cv2.resize(mask_inv, (earring_width,earring_height), interpolation = cv2.INTER_AREA)

    roi = image[y1:y2, x1:x2]
    
    roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    
    # roi_fg contains the orignal location of earring
    roi_fg = cv2.bitwise_and(earring,earring,mask = mask)
    
    # joining the roi_bg and roi_fg
    dst = cv2.add(roi_bg,roi_fg)
    # placing the joined image and saving to dst back over the original image
    image[y1:y2, x1:x2] = dst

    ############################################################### 
    ##########################Right Ear############################
    earring_height = hr
    earring_width = earring_height * org_width / org_height

    x1 = int(Xr - (earring_width/4))
    x2 = int(Xr + w + (earring_width/4))
    y1 = int(Yr + hr - (earring_height/2))
    y2 = int(Yr + hr + (earring_height/2))

    earring_width = int(x2 - x1)
    earring_height = int(y2 - y1)

    earring = cv2.resize(ear_img, (earring_width,earring_height), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (earring_width,earring_height), interpolation = cv2.INTER_AREA)
    mask_inv = cv2.resize(mask_inv, (earring_width,earring_height), interpolation = cv2.INTER_AREA)

    roi = image[y1:y2, x1:x2]

    roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # roi_fg contains the orignal location of earring
    roi_fg = cv2.bitwise_and(earring,earring,mask = mask)
        
    # joining the roi_bg and roi_fg
    dst = cv2.add(roi_bg,roi_fg)
        
    # placing the joined image and saving to dst back over the original image
    image[y1:y2, x1:x2] = dst
    ###############################################################

    

    cv2.imshow("Output", image)
    cv2.waitKey(0)




if __name__ == '__main__':
    main()

