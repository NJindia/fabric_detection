import cv2
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from scipy.stats import ks_2samp

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
    final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("bw",img_bin)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)
    # cv2.imshow('limg', limg)
    # cv2.imshow('CLAHE output', cl)
    # cv2.imshow('final', final)
    # cv2.waitKey(0)

    return final


def getImgs(parent_folder_path):
    imgs = []
    for f in listdir(parent_folder_path):
        if isfile(join(parent_folder_path, f)):
            path = join(parent_folder_path, f)
            img = cv2.imread(path, 1)
            imgs.append(Image(img , f, path))

    return imgs

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

def mean(hist):
    mean = 0.0
    for i in hist:
        mean += i
    mean = mean / len(hist)
    return mean

def getIntersection(hist_1, hist_2):
        """
        Calculates the common area between two histograms with the same # of bins
        :param hist_1: Histogram 1
        :param hist_2: Histogram 2
        :return: scalar between 0 and 1
        """
        minima = np.minimum(hist_1, hist_2)
        intersection = np.true_divide(np.sum(minima), np.sum(hist_2))

        return intersection

def getKStest(hist1, hist2):
    kStat, p = ks_2samp(hist1, hist2)
    return kStat, p

def bhattacharyya(hist1, hist2, bins):
    bhatt_coeff = 0
    for h1, h2 in zip(hist1, hist2):
        sqrt_of_product = h1*h2
        bhatt_coeff = bhatt_coeff + sqrt_of_product
    print("bhat_coeff: " + str(bhatt_coeff))
    hellinger = np.sqrt(1-bhatt_coeff)
    print("hellinger: " + str(hellinger))
    # hist1_mean = mean(hist1)
    # hist2_mean = mean(hist2)
    # p2 = 1/(np.sqrt(hist1_mean * hist2_mean * bins))
    # p3 = 0.0
    # for h1, h2 in zip(hist1, hist2):
    #     p3 = p3 + np.sqrt(h1 * h2)
    # p = 1 - (p2 * p3)
    # print(p)
    bhatta = -1 * np.log(bhatt_coeff)
    print("bhatta_dist: " + str(bhatta))
        
def getLChannel(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return l

def compareHist(img1, img2):
    # fileName = image1.fileName + '&&' + image2.fileName + '.csv'
    img1 = img1.image
    # img2 = img2.image
    np.set_printoptions(threshold=sys.maxsize)
    img1 = img1[1000:1100, 1200:1300]
    img2 = img2[1000:1100, 1200:1300]
    l1= getLChannel(img1)
    l2= getLChannel(img2)

    cv2.imwrite('images/smol/present.png', img1)
    cv2.imwrite('images/smol/not_present.png', img2)
    bins = 1000
    hist1 = hog(l1, orientations=bins, pixels_per_cell=(25, 25), cells_per_block=(1, 1))
    hist2 = hog(l2, orientations=bins, pixels_per_cell=(25, 25), cells_per_block=(1, 1))    
    # hists = np.append(hist1, hist2)
    # catArr = np.append(np.full(len(hist1), 'Present L'), np.full(len(hist2), 'Not Present L'))
    # arr = (catArr, hists)
    # arr = np.transpose(arr)
    # np.savetxt(fileName, arr, fmt='%s', delimiter=',', header="hist, value")

    # bhattacharyya(hist1, hist2, bins)
    # intersection = getIntersection(hist1, hist2)
    kStat, p = getKStest(hist1, hist2)

    return kStat, p



def main():
    i = 0
    # imgs = getImgs('images/dark_tests/dblue/outside')
    present = getImgs('images/darks')
    # avgPresentHist = getAvgHist(present)
    # refPic = cv2.imread('images/fabric_not_present/destacker_back_20201112_163305_WO0176_7377-BP_Inside.png', 1)
    refPic = cv2.imread('images/darks/db_10000.png', 1)
    # darkPic = cv2.imread('images/dark_contrast.png', 1)
    notPresent = getImgs('images/fabric_not_present')
    # avgNotPresentHist = getAvgHist(notPresent)
    print("present")
    for img in present:
        print(compareHist(img, refPic))
    print("not present")
    for img in notPresent:
        print(compareHist(img, refPic))
    # print(compareHist(darkPic, refPic))
if __name__ == '__main__':
    main()
