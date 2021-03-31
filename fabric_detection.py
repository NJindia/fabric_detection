import cv2
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from scipy.ndimage import rotate
from sklearn import svm
from joblib import load, dump
from datetime import datetime

class Image:
    def __init__(self, img=None, fileName=None, path=None, roi=None):
        self.image = img
        self.fileName = fileName
        self.path = path
        self.roi = roi

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
            roi = img[0:1544,400:1944]
            imgs.append(Image(img=img, fileName=f, path=path))
    return imgs

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

def getROI(img):
    center = (int(img.shape[1]/2), int(img.shape[0]/2)) #(x, y)
    x1 = center[0]-500
    x2 = center[0]+500
    y1 = center[1]-500
    y2 = center[1]+500
    # cv2.line(img, (x1, y1), (x2, y1), 1, 5)
    # cv2.line(img, (x1, y1), (x1, y2), 1, 5)
    # cv2.line(img, (x2, y1), (x2, y2), 1, 5)
    # cv2.line(img, (x1, y2), (x2, y2), 1, 5)
    return img[y1:y2,x1:x2]

def splitImg(img):
    imgs = []
    shape = img.shape #(y, x)
    y = 0
    x = 0
    while y < shape[0]:
        while x < shape[1]:
            im = img[y:y+100, x:x+100]
            imgs.append(im)
            x = x + 100
        y = y + 100
        x=0
    return imgs

def getRotations(img):
    rotations = []
    for i in range(0, 360, 15):
        print(i)
        rotation = rotate(img, i)
        print(i)
        
        roi = getROI(rotation)
        rotations.append(rotation)
    return rotations
      
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

def getSVM_CLF(present, notPresent):
    X = []
    y = []
    for image in present:
        rotations = getRotations(image.image)
        for rotation in rotations:
            split = splitImg(rotation)
            for img in split:
                X.append(rawImgToHist(img))
                y.append(1)
    for image in notPresent:
        for rotation in image.rotations:
            split = splitImg(rotation)
            for img in split:
                X.append(rawImgToHist(img))
                y.append(0)
    clf = svm.SVC()
    clf.fit(X, y)
    return clf

def makeCLF():
    start = datetime.now()
    print('start')
    present = getImages('images/SVM_training_present')
    notPresent = getImages('images/SVM_training_not_present')
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