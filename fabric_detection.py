import cv2
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
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

class FabricDetector:    
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

    def getImages(self, parent_folder_path):
        imgs = []
        for f in os.listdir(parent_folder_path):
            if os.path.isfile(os.path.join(parent_folder_path, f)):
                path = os.path.join(parent_folder_path, f)
                img = cv2.imread(path)
                roi = img[0:1544,374:1918]
                # imgEq = self.shiftHist(roi)
                imgs.append(Image(img=img, fileName=f, path=path, roi=roi))
        return imgs

    def getHOG(self, img):
        hist = hog(img, orientations=36, pixels_per_cell=(100, 100), cells_per_block=(1, 1))
        # cv2.imshow('hog', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return hist

    def getROI(self, img):
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

    def getRotations(self, img):
        rotations = []
        for i in range(0, 360, 15):
            image_center = tuple(np.array(img.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, i, 1.0)
            rotation = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)        
            roi = self.getROI(rotation)
            rotations.append(roi)
        return rotations
 
    def getLChannel(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return l

    def shiftHist(self, img):
        # cv2.imshow('og', img)
        start = datetime.now()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h[:,: ] =  50
        diff1 = 128 - cv2.mean(s)[0]
        diff2 = 128 - cv2.mean(v)[0] 
        s = s + int(diff1)
        v = v + int(diff2)
        s[s>255] = 255
        s[s<0] = 0
        v[v>255] = 255
        v[v<0] = 0
        s=s.astype(np.uint8)
        v=v.astype(np.uint8)
        hsvNew = cv2.merge((h, s, v))
        final = cv2.cvtColor(hsvNew, cv2.COLOR_HSV2BGR)
        # cv2.imshow('final', final)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return final


    def predictPresence(self, clf, image):
        sum = 0
        roi = self.getROI(image.roi)
        split = self.splitImg(roi)
        for im in split:
            hist = self.getHOG(im)
            hist = hist.reshape(1, -1)
            p = clf.predict(hist)
            if(p==1):
                sum += 1
        pred = sum/len(split)
        print('avg = ' + str(pred))
        if(pred > 0.5):
            return 1
        else:
            return 0

    def getSVM_CLF(self, present, notPresent):
        X = []
        y = []
        for image in present:
            # print(image.fileName)
            rotations = self.getRotations(image.roi)
            for rotation in rotations:
                split = self.splitImg(rotation)
                for img in split:
                    X.append(self.getHOG(img))
                    y.append(1)
        for image in notPresent:
            # print(image.fileName)
            rotations = self.getRotations(image.roi)
            for rotation in rotations:
                split = self.splitImg(rotation)
                for img in split:
                    X.append(self.getHOG(img))
                    y.append(0)
        clf = svm.SVC()
        print(len(y))
        clf.fit(X, y)
        return clf

    def makeCLF(self):
        start = datetime.now()
        print('start')
        present = self.getImages(os.path.join(self.package_dir,'images/SVM_training_present'))
        notPresent = self.getImages(os.path.join(self.package_dir, 'images/SVM_training_not_present'))
        getImgsTime = datetime.now()
        print('get Images time: ' + str(getImgsTime - start))
        clf = self.getSVM_CLF(present, notPresent)
        getCLFTime = datetime.now()
        print('get CLF time: ' + str(getCLFTime - getImgsTime))
        dump(clf, 'clf.pk1')
        return clf

    def readCLF(self):
        start = datetime.now()
        clf = load('clf.pk1')
        loadCLFTime = datetime.now()
        print('load CLF time: ' + str(loadCLFTime - start))

        return clf

    def sharpness(self, images):
        for image in images:
            im = self.getROI(image.roi)
            array = np.asarray(im, dtype=np.int32)
            print(np.diff(array,  axis=0).shape)
            dx = np.diff(array)[1:,:] # remove the first row
            dy = np.diff(array, axis=0)[:,1:] # remove the first column
            print(dy)
            dnorm = np.sqrt(dx**2 + dy**2)
            sharpness = np.average(dnorm)
            print(im.filename, sharpness)

    def __init__(self):
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        presImgs = self.getImages(os.path.join(self.package_dir,'images\SVM_test_present'))
        self.sharpness(presImgs)
        print('done')
        notPresImgs = self.getImages(os.path.join(self.package_dir,'images\SVM_test_not_present'))
        
        newCLF = True
        if newCLF:
            clf = self.makeCLF()
        else: 
            clf = self.readCLF()
        # np.savetxt('avgPresentHist.csv', avgPresentHist, fmt='%s', delimiter=',', header="value")
        # np.savetxt('avgNotPresentHist.csv', avgNotPresentHist, fmt='%s', delimiter=',', header="value")    
        print("present")
        for presImg in presImgs:
            presPrediction = self.predictPresence(clf, presImg)
            print(presImg.fileName + ' present prediction: ' + str(presPrediction))
        print("not present")
        for notPresImg in notPresImgs:
            notPresPrediction = self.predictPresence(clf, notPresImg)
            print(notPresImg.fileName + ' not present prediction: ' + str(notPresPrediction))
    



if __name__ == '__main__':
    fd = FabricDetector()