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
            img = cv2.imread(path)
            imgs.append(Image(img , f, path))

    return imgs

def getHOG(img):
    hist, vis= hog(img, orientations=360, pixels_per_cell=(25, 25), cells_per_block=(2, 2 ), visualize=True)
    cv2.imshow('hog', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return hist

def getsHOG(img):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(img,winStride,padding,locations)
    print(len(hist))
    return hist

def roi(img):
    return img[1000:1100, 1200:1300]

def getAvgHist(images, csvName):
    histArr = []
    csvArr = []
    for image in images:
        img = roi(image.image)
        # img = getLChannel(img)
        img = getSVChannels(img)
        cv2.imwrite('images/smol/' + image.fileName, img)
        hist = getHOG(img)
        histArr.append(hist)
        csvArr = np.append(csvArr, np.full(len(hist), image.fileName), axis=0)
    flatArr = [elem for hist in histArr for elem in hist]
    arr = (csvArr, flatArr)
    arr = np.transpose(arr)
    np.savetxt(csvName, arr, fmt='%s', delimiter=',', header="hist, value")
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
        
def getSVChannels(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h[:,: ] =  360 
    hsvNew = cv2.merge((h, s, v))

    final = cv2.cvtColor(hsvNew, cv2.COLOR_HSV2BGR)
    # final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('final', final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return final

def getLChannel(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return l

def compareHist(img, avgHist):
    fileName = img.fileName + '.csv'
    img = img.image
    img = roi(img)
    # img = getLChannel(img)
    img = getSVChannels(img)
    hist = getHOG(img)
    intersection = getIntersection(hist, avgHist)
    return intersection

def main():
    i = 0
    # imgs = getImgs('images/dark_tests/dblue/outside')
    present = getImgs('images/darks')
    notPresent = getImgs('images/fabric_not_present')
    avgPresentHist = getAvgHist(present, 'presentImgs.csv')
    avgNotPresentHist = getAvgHist(notPresent, 'notPresentImgs.csv')
    np.savetxt('avgPresentHist.csv', avgPresentHist, fmt='%s', delimiter=',', header="value")
    np.savetxt('avgNotPresentHist.csv', avgNotPresentHist, fmt='%s', delimiter=',', header="value")    
    print("present")
    for img in present:
        print(compareHist(img, avgPresentHist))
    print("not present")
    for img in notPresent:
        print(compareHist(img, avgPresentHist))


if __name__ == '__main__':
    main()
