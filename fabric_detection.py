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

def getHOG(img):
    hist = hog(img, orientations=36, pixels_per_cell=(50, 50), cells_per_block=(1, 1))
    return hist

def roi(img):
    return img[1000:1050, 1200:1250]

def getAvgHist(images):
    histArr = []
    csvArr = []
    for image in images:
        img = roi(image.image)
        l = getLChannel(img)
        cv2.imwrite('images/smol/' + image.fileName, l)
        hist = getHOG(l)
        np.savetxt(image.fileName + '.csv', hist, fmt='%s', delimiter=',', header="value")
        histArr.append(hist)
        csvArr = np.append(csvArr, np.full(len(hist), image.fileName), axis=0)
        print(len(hist))
    flatArr = [elem for hist in histArr for elem in hist]
    print(flatArr)
    arr = (csvArr, flatArr)
    arr = np.transpose(arr)
    np.savetxt('imgs.csv', arr, fmt='%s', delimiter=',', header="hist, value")
    sum = np.full(len(histArr[0]), 0)
    avg = []
    for hist in histArr:
        sum = sum + hist
    avg = sum / len(histArr)
    return avg


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
        
def getLChannel(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return l

def compareHist(img1, avgHist):
    fileName = img1.fileName + '.csv'
    img1 = img1.image
    # img2 = img2.image
    np.set_printoptions(threshold=sys.maxsize)
    img1 = roi(img1)
    img2 = roi(img2)
    l1= getLChannel(img1)
    l2= getLChannel(img2)

    cv2.imwrite('images/smol/present.png', img1)
    cv2.imwrite('images/smol/not_present.png', img2)
    hist1, imgH = getHOG(l1)
    hist2, imgH2 = getHOG(l2)
    cv2.imshow('imgH', imgH)
    cv2.imshow('imgH2', imgH2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    hists = np.append(hist1, hist2)
    catArr = np.append(np.full(len(hist1), 'Present L'), np.full(len(hist2), 'Not Present L'))
    arr = (catArr, hists)
    arr = np.transpose(arr)
    np.savetxt(fileName, arr, fmt='%s', delimiter=',', header="hist, value")

    # bhattacharyya(hist1, hist2, bins)
    # intersection = getIntersection(hist1, hist2)
    kStat, p = getKStest(hist1, hist2)

    return kStat, p


def main():
    i = 0
    # imgs = getImgs('images/dark_tests/dblue/outside')
    # present = getImgs('images/darks')
    notPresent = getImgs('images/fabric_not_present')

    avgPresentHist = getAvgHist(notPresent)
    np.savetxt('avgHist.csv', avgPresentHist, fmt='%s', delimiter=',', header="value")
    print('checkpoint')
    
    # # refPic = cv2.imread('images/fabric_not_present/destacker_back_20201112_163305_WO0176_7377-BP_Inside.png', 1)
    # refPic = cv2.imread('images/darks/db_10000.png', 1)
    # # darkPic = cv2.imread('images/dark_contrast.png', 1)
    # notPresent = getImgs('images/fabric_not_present')
    # # avgNotPresentHist = getAvgHist(notPresent)
    
    # print("present")
    # for img in present:
    #     print(compareHist(img, refPic))
    # print("not present")
    # for img in notPresent:
    #     print(compareHist(img, refPic))
if __name__ == '__main__':
    main()
