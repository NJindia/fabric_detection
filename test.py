import numpy as np
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
                img = cv2.imread(path)
                roi = img[0:1544,374:1918]
                # imgEq = self.shiftHist(roi)
                imgEq = roi
                image = Image(img=imgEq, fileName=f, path=path)
                # acutance = self.getAcutance(image)
                # print('OG: ', f, acutance)
                # acutance = self.getAcutance(imgEq)
                # print('SHIFT: ', f, acutance)
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

    def increaseContrast(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=1000.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final


    def detectLines(self, img):
        img = self.increaseContrast(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
                    angles.append(angle)
        if lines is not None:
            # lines = self.checkLines(lines, angles)
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.imshow('lines', line_image)
        cv2.waitKey(0)
        return lines

    def checkLines(self, lines, angles):
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
        print(len(lines))
        print(len(filteredLines))
        return filteredLines

    def __init__(self):
        package_dir = os.path.dirname(os.path.abspath(__file__))

        # self.readCLFandPCA()
        # asians = self.getImages(os.path.join(package_dir,'images/SVM_test_present'))
        # notPres = self.getImages(os.path.join(package_dir,'images/SVM_test_not_present'))
        # dist = self.getDist(asians)
        # print(dist)
        # dist = self.getDist(notPres)
        # print(dist)
        pres = self.getImages(os.path.join(package_dir,'images/SVM_training_present'))
        notPres = self.getImages(os.path.join(package_dir,'images/SVM_training_not_present'))
        for image in pres:
            img = self.getROI(image.img)
            lines = self.detectLines(img)
            print(image.fileName, lines)
        print('not pres')
        for image in notPres:
            img = self.getROI(image.img)

            lines = self.detectLines(image)
            print(image.fileName, lines)
if __name__ == '__main__':
    fd = FabricDetector()