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
from sklearn import svm

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
            temp = splitImg(img)
            [imgs.append(i) for i in temp]
    return imgs


# def getImgs(parent_folder_path):
#     imgs = []
#     for f in listdir(parent_folder_path):
#         if isfile(join(parent_folder_path, f)):
#             path = join(parent_folder_path, f)
#             img = cv2.imread(path)
#             imgs.append(Image(img , f, path))
#     return imgs

def getSVM_CLF(present, notPresent):
    X = []
    y = []
    for img in present:
        X.append(rawImgToHist(img))
        y.append(1)
    for img in notPresent:
        X.append(rawImgToHist(img))
        y.append(0)
    # np.savetxt(csvName, arr, fmt='%s', delimiter=',', header="hist, value")
    clf = svm.SVC()
    clf.fit(X, y)
    return clf

def rawImgToHist(img):
    # img = getLChannel(img)
    img = getSVChannels(img)
    hist = getHOG(img)
    return hist

def getHOG(img):
    hist, vis= hog(img, orientations=36, pixels_per_cell=(25, 25), cells_per_block=(2, 2), visualize=True)
    # cv2.imshow('hog', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return hist

def roi(img):
    return img[1000:1100, 1200:1300]

def splitImg(img):
    imgs = []
    y = 200
    x = 500
    while y <= 1300:
        while x <=1600:
            im = img[y:y+100, x:x+100]
            imgs.append(im)
            x = x + 100
        y = y  + 100
        x=500

    return imgs

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
    return avg, histArr

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

def predictPresence(clf, img):
    sum = 0
    imgParts = splitImg(img)
    for i in imgParts:
        hist = rawImgToHist(i)
        hist = hist.reshape(1, -1)
        p = clf.predict(hist)
        print(p)
        if(p==1):
            sum += 1
    pred = sum/len(imgParts)
    if(pred > 0.5):
        return 1
    else:
        return 0

def main():
    i = 0
    # imgs = getImgs('images/dark_tests/dblue/outside')
    presImg = cv2.imread('images/SVM_test/present.png')
    notPresImg = cv2.imread('images/SVM_test/no_shirt3.png')
    present = getImgs('images/darks')
    notPresent = getImgs('images/fabric_not_present')
    clf = getSVM_CLF(present, notPresent)
    # np.savetxt('avgPresentHist.csv', avgPresentHist, fmt='%s', delimiter=',', header="value")
    # np.savetxt('avgNotPresentHist.csv', avgNotPresentHist, fmt='%s', delimiter=',', header="value")    
    print("present")
    presPrediction = predictPresence(clf, presImg)
    print("not present")
    notPresPrediction = predictPresence(clf, notPresImg)

    print('present prediction: ' + str(presPrediction))
    print('not present prediction: ' + str(notPresPrediction))
    





if __name__ == '__main__':
    main()
