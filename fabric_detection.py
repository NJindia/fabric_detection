import cv2
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from sklearn import svm
from joblib import load, dump
from datetime import datetime

class Image:
    def __init__(self, img=None, fileName=None, path=None, split=None):
        self.image = img
        self.fileName = fileName
        self.path = path
        self.split = split

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

def getImages(parent_folder_path):
    imgs = []
    for f in listdir(parent_folder_path):
        if isfile(join(parent_folder_path, f)):
            path = join(parent_folder_path, f)
            img = cv2.imread(path)
            split = splitImg(img)
            imgs.append(Image(img, f, path, split))
    return imgs

def getSVM_CLF(present, notPresent):
    X = []
    y = []
    for image in present:
        for img in image.split:
            X.append(rawImgToHist(img))
            y.append(1)
    for image in notPresent:
        for img in image.split:
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
    hist = hog(img, orientations=36, pixels_per_cell=(100, 100), cells_per_block=(1, 1))
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

def predictPresence(clf, img):
    sum = 0
    for i in img.split:
        hist = rawImgToHist(i)
        hist = hist.reshape(1, -1)
        p = clf.predict(hist)
        if(p==1):
            sum += 1
    pred = sum/len(img.split)
    print('avg = ' + str(pred))
    if(pred > 0.5):
        return 1
    else:
        return 0

def makeCLF():
    start = datetime.now()
    present = getImages('images/darks')
    notPresent = getImages('images/fabric_not_present')
    getImgsTime = datetime.now()
    print('get Images time: ' + str(getImgsTime - start))
    clf = getSVM_CLF(present, notPresent)
    getCLFTime = datetime.now()
    print('get CLF time: ' + str(getCLFTime - getImgsTime))
    dump(clf, 'clf.pk1')
    return clf

def readCLF():
    start = datetime.now()
    clf = load('clf.pk1')
    loadCLFTime = datetime.now()
    print('load CLF time: ' + str(loadCLFTime - start))

    return clf

def main():
    presImgs = getImages('images/SVM_test_present')
    notPresImgs = getImages('images/SVM_test_not_present')
    newCLF = True
    if newCLF:
        clf = makeCLF()
    else: 
        clf = readCLF()
    # np.savetxt('avgPresentHist.csv', avgPresentHist, fmt='%s', delimiter=',', header="value")
    # np.savetxt('avgNotPresentHist.csv', avgNotPresentHist, fmt='%s', delimiter=',', header="value")    
    print("present")
    for presImg in presImgs:
        presPrediction = predictPresence(clf, presImg)
        print(presImg.fileName + ' present prediction: ' + str(presPrediction))
    print("not present")
    for notPresImg in notPresImgs:
        notPresPrediction = predictPresence(clf, notPresImg)
        print(notPresImg.fileName + ' not present prediction: ' + str(notPresPrediction))
    
if __name__ == '__main__':
    main()
