import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


class Image:
    def __init__(self, img, prev, fileName, path):
        self.currValue = img
        self.prevValue = prev
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


def blur(img, kernel_size=17):
    gaussian = cv2.GaussianBlur(
        img, (kernel_size, kernel_size), 0)  # GOOD! Reduces noise
    return gaussian


def sharpen(img, kernel_size=17):
    #    gaussian = blur(img, kernel_size)
    #    sharpen = cv2.addWeighted(gaussian, 1.5, img, -.5, 0)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    return img

    #low_threshold = 50
    #high_threshold = 150

    #filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #filter = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
    #sharpen = cv2.filter2D(gaussian, -1, filter)


def drawLines(img):
    edges = cv2.Canny(img,25,125,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    return img, edges


def drawLinesP(
    img, low_threshold=50, high_threshold=150, rho=1, theta=np.pi / 180, threshold=10, min_line_length=100, max_line_gap = 7
    ):
    edges = cv2.Canny(img,25,125,apertureSize = 3)

    # distance resolution in pixels of the Hough grid
    # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    # minimum number of pixels making up a line
    # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            np.array([]), min_line_length, max_line_gap)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(edges, 0.8, line_image, 1, 0)
    return lines_edges, edges


def getImgs(parent_folder_path):
    imgs = []
    for f in listdir(parent_folder_path):
        if isfile(join(parent_folder_path, f)):
            path = join(parent_folder_path, f)
            img = cv2.imread(path, 1)
            roi = img[195:1390, 480:1675]
            imgs.append(Image(roi, img, f, path))
    return imgs


def binary(img):
    bitwise = cv2.bitwise_not(img)
    return bitwise


def main():
    darks = getImgs('images/darks')
    nps = getImgs('images/fabric_not_present')
    for i in range(0, 7):
        img = darks[i]
        nofab = nps[i]

        original = img.prevValue
        img.currValue = increaseContrast(img.currValue)

        fabOG = nofab.prevValue
        nofab.currValue = increaseContrast(nofab.currValue)

        img.currValue = cv2.cvtColor(img.currValue, cv2.COLOR_BGR2GRAY)
        nofab.currValue = cv2.cvtColor(nofab.currValue, cv2.COLOR_BGR2GRAY)
        
        plt.subplot(1, 2, 1), plt.imshow(img.currValue, cmap='gray')
        plt.title('original fabric'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(nofab.currValue, cmap='gray')
        plt.title('original no fabric'), plt.xticks([]), plt.yticks([])
 
        plt.ion()
        plt.show()

        args = [None]
        while(args[0] != 'next'):
            cmd = (
                input("Enter command (blur #, sharpen #, draw_lines # #, undo, next):")).lower()
            args = cmd.split(" ")
            cv2.destroyAllWindows()
            if args[0] == 'blur':
                img.prevValue = img.currValue
                if len(args) > 1:
                    img.currValue = blur(img.currValue, int(args[1]))
                    nofab.currValue = blur(nofab.currValue, int(args[1]))
                else:
                    img.currValue = blur(img.currValue)
                    nofab.currValue = blur(nofab.currValue)
                plt.subplot(2, 2, 1), plt.imshow(img.prevValue, cmap='gray')
                plt.title('previous fabric'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, 2, 2), plt.imshow(img.currValue, cmap='gray')
                plt.title(cmd), plt.xticks([]), plt.yticks([])
                
                plt.subplot(2, 2, 3), plt.imshow(nofab.prevValue, cmap='gray')
                plt.title('previous no_fabric'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, 2, 4), plt.imshow(nofab.currValue, cmap='gray')
                plt.title(cmd), plt.xticks([]), plt.yticks([])
                
                plt.ion()
                plt.show()
            elif args[0] == 'undo':
                img.currValue = img.prevValue
                img.prevValue = original
                plt.imshow(img.currValue, cmap='gray')
                plt.title(cmd), plt.xticks([]), plt.yticks([])
                plt.ion()
                plt.show()
            elif args[0] == 'draw_lines':
                lines, edges = drawLines(img.currValue)
                plt.subplot(2, 3, 1), plt.imshow(img.currValue, cmap='gray')
                plt.title('previous'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, 3, 2), plt.imshow(edges, cmap='gray')
                plt.title('edges'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, 3, 3), plt.imshow(lines, cmap='gray')
                plt.title('lines'), plt.xticks([]), plt.yticks([])

                lines2, edges2 = drawLines(nofab.currValue)
                plt.subplot(2, 3, 4), plt.imshow(nofab.currValue, cmap='gray')
                plt.title('previous'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, 3, 5), plt.imshow(edges2, cmap='gray')
                plt.title('edges'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, 3, 6), plt.imshow(lines2, cmap='gray')
                plt.title('lines'), plt.xticks([]), plt.yticks([])
                plt.ion()
                plt.show()
            elif args[0] == 'draw_lines_p':
                if(len(args) > 1): 
                    args.pop(0)
                    lines, edges = drawLinesP(img.currValue, *args)
                    lines2, edges2 = drawLinesP(nofab.currValue, *args)
                else:
                    lines, edges = drawLinesP(img.currValue)
                    lines2, edges2 = drawLinesP(nofab.currValue)
                plt.subplot(2, 3, 1), plt.imshow(img.currValue, cmap='gray')
                plt.title('previous no fab'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, 3, 2), plt.imshow(edges, cmap='gray')
                plt.title('edges'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, 3, 3), plt.imshow(lines, cmap='gray')
                plt.title('lines'), plt.xticks([]), plt.yticks([])

                plt.subplot(2, 3, 4), plt.imshow(nofab.currValue, cmap='gray')
                plt.title('previous no fab'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, 3, 5), plt.imshow(edges2, cmap='gray')
                plt.title('edges'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, 3, 6), plt.imshow(lines2, cmap='gray')
                plt.title('lines'), plt.xticks([]), plt.yticks([])
                plt.ion()
                plt.show()

                plt.ion()
                plt.show()
            elif args[0] == 'sharpen':
                img.prevValue = img.currValue
                if len(args) > 1:
                    img.currValue = sharpen(img.currValue, int(args[1]))
                else:
                    img.currValue = sharpen(img.currValue)
                plt.subplot(1, 2, 1), plt.imshow(img.prevValue, cmap='gray')
                plt.title('previous'), plt.xticks([]), plt.yticks([])
                plt.subplot(1, 2, 2), plt.imshow(img.currValue, cmap='gray')
                plt.title(cmd), plt.xticks([]), plt.yticks([])
                plt.ion()
                plt.show()
            elif args[0] == 'binary':
                img.prevValue = img.currValue
                img.currValue = binary(img.currValue)
                plt.subplot(1, 2, 1), plt.imshow(img.prevValue, cmap='gray')
                plt.title('previous'), plt.xticks([]), plt.yticks([])
                plt.subplot(1, 2, 2), plt.imshow(img.currValue, cmap='gray')
                plt.title(cmd), plt.xticks([]), plt.yticks([])
                plt.ion()
                plt.show()


if __name__ == '__main__':
    main()
