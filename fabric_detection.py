import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


class Image:
    def __init__(self, img, fileName, path):
        self.image = img
        self.fileName = fileName
        self.path = path

    def __str__(self):
        return self.fileName


def fabricDetection(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def increaseContrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=1000.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cv2.imshow("bw",img_bin)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)
    # cv2.imshow('limg', limg)
    # cv2.imshow('CLAHE output', cl)
    # cv2.imshow('final', final)
    # cv2.waitKey(0)

    return final


def roi(img):
    roi = img[480:1675, 195:1390]
    return

def drawLines(img, low_threshold, high_threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # for kernel_size in [19, 21]:
    kernel_size = 19
    gaussian = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)  # GOOD! Reduces noise
    #low_threshold = 50
    #high_threshold = 150

    edges = cv2.Canny(gaussian, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(gaussian) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    if lines is not None:
       for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(edges, 0.8, line_image, 1, 0)
    return lines_edges, gaussian

def detectLines(image):
    img = increaseContrast(image.image)
    edges = drawLines(img, 50, 150)
    edges2= drawLines(img, 75, 125)
    edges3= drawLines(img, 100, 150)

    fig = plt.figure()
    plt.subplot(2, 2, 1),plt.imshow(img,cmap = 'gray')
    plt.title('edge1'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3),plt.imshow(edges2,cmap = 'gray')
    plt.title('Edge2 Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4),plt.imshow(edges3,cmap = 'gray')
    plt.title('Edge3 Image'), plt.xticks([]), plt.yticks([])


    plt.show()

    #cv2.imshow(image.fileName, image.image)
    #cv2.imshow('contrasted ' + image.fileName, gray)
    #cv2.imshow('lines  ' + image.fileName, lines_edges)


    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
        

    #return lines


def getImgs(parent_folder_path):
    imgs = []
    for f in listdir(parent_folder_path):
        if isfile(join(parent_folder_path, f)):
            path = join(parent_folder_path, f)
            img = cv2.imread(path, 1)
            roi = img[195:1390, 480:1675]
            imgs.append(Image(roi, f, path))

    return imgs

def checkDark():
    image = cv2.imread('images/contrast_black.png')
    img = increaseContrast(image)
    bitwise = cv2.bitwise_not(img)
    lines, g1 = drawLines(img, 50, 150)
    lines2, g2 = drawLines(bitwise, 50, 150)

    fig = plt.figure()
    plt.subplot(2, 3, 1),plt.imshow(img,cmap = 'gray')
    plt.title('img'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(2, 3, 2),plt.imshow(g1,cmap = 'gray')
    plt.title('gaussian'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 3),plt.imshow(lines,cmap = 'gray')
    plt.title('Lines Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(2, 3, 4),plt.imshow(bitwise,cmap = 'gray')
    plt.title('bitwise not'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(2, 3, 5),plt.imshow(g2,cmap = 'gray')
    plt.title('gaussian'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 6),plt.imshow(lines2,cmap = 'gray')
    plt.title('Lines Image'), plt.xticks([]), plt.yticks([])

    plt.show()


def main():
    i = 0

    checkDark()

    imgs = getImgs('images/test_images')

    print(len(imgs))

    for img in imgs:
        i = i + 1
        lines = detectLines(img)
        if lines is not None and len(lines > 1):
            cv2.imwrite('images/processed/present/' +
                        str(i) + '_' + img.fileName, img.image)
        else:
            cv2.imwrite('images/processed/not_present/' +
                        str(i) + '_' + img.fileName, img.image)


if __name__ == '__main__':
    main()
