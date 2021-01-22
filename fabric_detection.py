import cv2 
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


def fabricDetection(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def increaseContrastGetBinary(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=1000.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    (thresh, img_bin) = cv2.threshold(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow("bw",img_bin)
    #cv2.imshow('l_channel', l)
    #cv2.imshow('a_channel', a)
    #cv2.imshow('b_channel', b)
    #cv2.imshow('limg', limg)
    #cv2.imshow('CLAHE output', cl)
    #cv2.imshow('final', final)
    #cv2.waitKey(0)

    return final, img_bin


def detectLines(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for kernel_size in [17, 19, 21]:
        gaussian = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0) #GOOD! Reduces noise
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(gaussian, low_threshold, high_threshold)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 100  # minimum number of pixels making up a line
        max_line_gap = 10  # maximum gap in pixels between connectable line segments
        line_image = np.copy(gaussian) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            
            # Draw the lines on the  image
            lines_edges = cv2.addWeighted(edges, 0.8, line_image, 1, 0)

            #cv2.imshow(name, img)
            #cv2.imshow(name + "_edges", edges)
            cv2.imshow(str(kernel_size) + "_gaussian_lines.png", lines_edges)

    #median = cv2.medianBlur(gray, kernel_size)
    #bilateral = cv2.bilateralFilter(gray, kernel_size, 200, 200)    
    #compareEdges("median", median)
    #compareEdges("bilateral", bilateral)
    #cv2.imshow("gray", gray)
    #cv2.imshow("edge image", lines_edges)
    #cv2.imwrite("images/processed/gray.png", gray)
    #cv2.imwrite("images/processed/gaussian.png", gaussian)
    #cv2.imwrite("images/processed/edges.png", edges)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getImgs(parent_folder_path):
    paths = [join(parent_folder_path, f) for f in listdir(parent_folder_path) if isfile(join(parent_folder_path, f))]
    imgs = [cv2.imread(path, 1) for path in paths]
    return imgs



def main():
    #cv2.namedWindow('back',cv2.WINDOW_NORMAL)
    #cv2.namedWindow('front',cv2.WINDOW_NORMAL)
    #cv2.namedWindow('original',cv2.WINDOW_NORMAL)
    #cv2.namedWindow('contrasted',cv2.WINDOW_NORMAL)

    imgs = getImgs('images/fabric_not_present')
    imgs = getImgs('images/fabric_present')

    for img in imgs: 
        (imgc, img_bin) =increaseContrastGetBinary(img)
        #cv2.imwrite("binary.png", img_bin)
        detectLines(imgc)


    


    



if __name__=='__main__':
    main()