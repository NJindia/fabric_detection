import numpy as np
import math
import os
import matplotlib as plt
import cv2
from joblib import load, dump
from sklearn import svm
from sklearn.decomposition import PCA
from skimage.feature import hog, greycomatrix, greycoprops



class Image:
    def __init__(self, img=None, fileName=None, path=None):
        self.img = img
        self.fileName = fileName
        self.path = path

    def __str__(self):
        return self.fileName
    def __repr__(self):
        return self.fileName

class FabricDetector:
    def getImages(self, parent_folder_path):
        imgs = []
        for f in os.listdir(parent_folder_path):
            if os.path.isfile(os.path.join(parent_folder_path, f)):
                path = os.path.join(parent_folder_path, f)
                img = cv2.imread(path, 0)
                roi = img[0:1544,374:1918]
                imgEq = roi
                image = Image(img=imgEq, fileName=f, path=path)
                imgs.append(image)
        return imgs

    def getAvgHist(self, hists):
        sum = np.full(len(hists[0]), 0)
        for hist in hists:
            sum = np.add(sum, hist)
        divisor = np.full(len(hists[0]), len(hists))
        average = np.divide(sum, divisor)
        return average

    def getROI(self, img):
        center = (int(img.shape[1]/2), int(img.shape[0]/2)) #(x, y)
        x1 = center[0]-500
        x2 = center[0]+500
        y1 = center[1]-500
        y2 = center[1]+500
        return img[y1:y2,x1:x2]

    def splitImg(self, img):
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

    def getHOG(self, img):
        hist = hog(img, orientations=36, pixels_per_cell=(100, 100), cells_per_block=(1, 1))
        return hist

    def readCLFandPCA(self):
        self.clf = load('clf.pk1')
        self.pca = load('pca.pk1')

    def getDist(self, images):
        avgHists = []
        for image in images:
            sum = 0
            pred = 0
            hists = []
            roi = self.getROI(image.img)
            split = self.splitImg(roi)            
            for img in split:
                hist = self.getHOG(img)
                hists.append(hist)
            avgHist = self.getAvgHist(hists)
            avgHists.append(avgHist)
        avgHistsPCA = self.pca.transform(avgHists)
        dist = self.clf.decision_function(avgHistsPCA)
        return dist

    def scatterPlot(self):
        arr = np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]])
    
    def GLRLM(self, images):
        pass

    def __init__(self):
        package_dir = os.path.dirname(os.path.abspath(__file__))

        # self.readCLFandPCA()
        # asians = self.getImages(os.path.join(package_dir,'images/SVM_test_present'))
        # notPres = self.getImages(os.path.join(package_dir,'images/SVM_test_not_present'))
        # dist = self.getDist(asians)
        # print(dist)
        # dist = self.getDist(notPres)
        # print(dist)
        angles = []
        for i in range(0, 360, 15):
            angles.append(i*(math.pi/180)) 
        pres = self.getImages(os.path.join(package_dir,'images/SVM_training_present'))
        notPres = self.getImages(os.path.join(package_dir,'images/SVM_training_not_present'))
        for image in pres:
            img = self.getROI(image.img)
            result = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256)
            res2 = greycoprops(result, prop='dissimilarity')
            sum = np.average(res2)
            print(image.fileName, sum, res2)
        print('not pres')
        for image in notPres:
            img = self.getROI(image.img)
            result = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256)
            res2 = greycoprops(result, prop='dissimilarity')
            sum = np.average(res2)
            print(image.fileName, sum, res2)
if __name__ == '__main__':
    fd = FabricDetector()