import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure


class Image:
    def __init__(self, img, fileName, path):
        self.image = img
        self.fileName = fileName
        self.path = path

    def __str__(self):
        return self.fileName


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


def detectLines(image):
    img = increaseContrast(image.image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow(image.fileName, gray)
    cv2.waitKey(0)

    kernel_size = 19
    gaussian = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)  # GOOD! Reduces noise
    low_threshold = 50
    high_threshold = 150

    edges = cv2.Canny(gaussian, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid #####TODO? CHECK IF YOU CAN GO IN A CERTAIN ORIENTATION
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 150  # minimum number of pixels making up a line
    max_line_gap = 15  # maximum gap in pixels between connectable line segments
    line_image = np.copy(gaussian) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            np.array([]), min_line_length, max_line_gap)
    angles = []

    # BLACK MAGIC BREAKS CODE DO NOT COMBINE
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = np.rad2deg(np.arctan2(y1 - y2, x1 - x2))
                print(angle)
                angles.append(angle)
    if lines is not None:
        lines = checkLines(lines, angles)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(edges, 0.8, line_image, 1, 0)
    # edges2= drawLines(img, 75, 125)
    # edges3= drawLines(img, 100, 150)

    hist(edges)
    
    fig = plt.figure()
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('edge1'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(lines_edges, cmap='gray')
    plt.title('lines Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(gray, cmap='gray')
    plt.title('gray Image'), plt.xticks([]), plt.yticks([])

    plt.show()
    

    # cv2.imshow(image.fileName, image.image)
    # cv2.imshow('contrasted ' + image.fileName, gray)
    # cv2.imshow('lines  ' + image.fileName, lines_edges)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return lines


def getImgs(parent_folder_path):
    imgs = []
    for f in listdir(parent_folder_path):
        if isfile(join(parent_folder_path, f)):
            path = join(parent_folder_path, f)
            img = cv2.imread(path, 1)
            roi = img[195:1390, 480:1675]
            imgs.append(Image(roi, f, path))

    return imgs


def checkLines(lines, angles):
    filteredLines = []
    i = 0

    for angle in angles:
        good_angles = 0
        for angle2 in angles:
            if abs(angle2 - angle) < 30:  # 30 DEG THRESHOLD
                good_angles = good_angles+1
        if(good_angles/len(angles) > .6 ):  # 60% THRESHOLD
            filteredLines.append(lines[i])
        i = i+1
    return filteredLines


def hist(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True)
    for e in fd:
        print(e)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()

def getAvgHist(images):
    histArr = []
    for image in images:
        img = increaseContrast(image.image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel_size = 19
        gaussian = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)  # GOOD! Reduces noise
      
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(gaussian, low_threshold, high_threshold)
        hist, hog_image = hog(edges, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
        histArr.append(hist)
        print(hist)
    sum = [0]
    avg = []
    for hist in histArr:
        pass
    return avg

def compareHist():
    np.set_printoptions(threshold=sys.maxsize)
    img1 = cv2.imread('images/not_present_img.png', 1)
    img2 = cv2.imread('images/image_10000.png')
    img1 = img1[195:1390, 480:1675]
    img2 = img2[195:1390, 480:1675]
    img1 = increaseContrast(img1)
    img2 = increaseContrast(img2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kernel_size = 19
    gaussian1 = cv2.GaussianBlur(gray1, (kernel_size, kernel_size), 0)
    gaussian2 = cv2.GaussianBlur(gray2, (kernel_size, kernel_size), 0)
    
    low_threshold = 50
    high_threshold = 150
    edges1 = cv2.Canny(gaussian1, low_threshold, high_threshold)
    edges2 = cv2.Canny(gaussian2, low_threshold, high_threshold)
    hist1 = hog(edges1, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    hist2 = hog(edges2, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    fo = open("fabric_present_avg_hist.txt", "w+")
    fo.write(str(hist2))
    fo = open("fabric_not_present_avg_hist.txt", "w+")
    fo.write(str(hist1))
    fo.close()



def main():
    compareHist()
    # i = 0
    # # imgs = getImgs('images/dark_tests/dblue/outside')
    # present = getImgs('images/fabric_present')
    # avgPresentHist = getAvgHist(present)
    
    # # notPresent = getImgs('images/fabric_not_present')
    # # avgNotPresentHist = getAvgHist(notPresent)
    
    # print(avgPresentHist)
    # fo = open("fabric_present_avg_hist.txt", "w+")
    # fo.write(avgPresentHist)
    # # fo = open("fabric_not_present_avg_hist.txt", "w+")
    # # fo.write(avgNotPresentHist)
    # fo.close()
    
    
    # for img in imgs:
    #     i = i + 1
    #     lines = detectLines(img)
        # if lines is not None and len(lines) > 5:
        #     print(str(i) + ':' + str(len(lines)))
        #     cv2.imwrite('images/processed/present/' +
        #                 str(i) + '_' + img.fileName, img.image)
        # else:
        #     if lines is None:
        #         print(str(i) + ':0')
        #     else:
        #         print(str(i) + ':' + str(len(lines)))

        #     cv2.imwrite('images/processed/not_present/' +
        #                 str(i) + '_' + img.fileName, img.image)


if __name__ == '__main__':
    main()
