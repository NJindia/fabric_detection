
import pickle
from sys import path
import cv2
from time import time
from mahotas import features
import numpy as np
import os
from matplotlib import pyplot as plt
from numpy.lib.polynomial import poly
from skimage import feature, exposure
from skimage.feature import hog, greycoprops, greycomatrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_validate
from joblib import load, dump
import mahotas as mt
from datetime import datetime

## Simple class to pair image data to filename and path
class Image:
    def __init__(self, img=None, filename=None, path=None):
        self.img = img
        self.filename = filename
        self.path = path

    def __str__(self):
        return self.filename
    def __repr__(self):
        return self.filename

class FabricDetector:  
    # Gets any images from the   
    def get_images(self, parent_folder_path):
        imgs = []
        for f in os.listdir(parent_folder_path):
            if os.path.isfile(os.path.join(parent_folder_path, f)):
                path = os.path.join(parent_folder_path, f)
                img = cv2.imread(path)
                roi = img[0:1544,374:1918]
                img_eq = roi
                image = Image(img=img_eq, filename=f, path=path)
                imgs.append(image)
        return imgs

    def get_hog(self, img):
        hist = hog(img, orientations=36, pixels_per_cell=(100, 100), cells_per_block=(1, 1))
        return hist

    def get_roi(self, img):
        center = (int(img.shape[1]/2), int(img.shape[0]/2)) #(x, y)
        x1 = center[0]-500
        x2 = center[0]+500
        y1 = center[1]-500
        y2 = center[1]+500
        return img[y1:y2,x1:x2]

    def split_img(self, img):
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

    def get_rotations(self, img):
        rotations = []
        for i in range(0, 360, 15):
            image_center = tuple(np.array(img.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, i, 1.0)
            rotation = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            roi = self.get_roi(rotation)
            rotations.append(roi)
        return rotations
        
    def get_glcm_features(self, grey):
        grey_int = np.uint8(255*grey)
        gcm = greycomatrix(grey_int, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
        glcm = greycoprops(gcm, prop='dissimilarity')[0]
        return glcm
    
    def equalize_hist(self, img, clip_limit=.5):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey_eq = exposure.equalize_adapthist(grey, clip_limit=clip_limit)
        return grey_eq

    def predict_presence(self, images):
        predictions = []
        for image in images:
            pred = 0
            hists = []
            roi = self.get_roi(image.img)
            grey_eq = self.equalize_hist(roi)
            split = self.split_img(grey_eq)            
            for img in split:
                hist = self.get_hog(img)
                hists.append(hist)
            avgHist = self.getAvgHist(hists)
            glcm = self.get_glcm_features(grey_eq)
            glm_normalized = self.normalizeArr(glcm, min=self.GLCMmin, max = self.GLCMmax)
            avg_hist_normalized = self.normalizeArr(avgHist, min=self.HOGmin, max=self.HOGmax)
            trainArr = np.concatenate((avg_hist_normalized, glm_normalized))
            # trainArr = avgHistNormalized
            # trainArr = gcpNormalized
            
            pca = self.pca.transform(trainArr.reshape(1, -1))
            pred = self.clf.predict(pca)
            predictions.append([image.filename, pred])

            #DEBUG
            dist = self.clf.decision_function(pca)
            self.dists[image.filename] = dist
            print(image.filename, dist)
        return predictions

    def make_clf_and_pca(self, present, notPresent):
        start = datetime.now()
        trainData = []
        for image in present:
            rotations = self.get_rotations(image.img)
            for rotation in rotations:
                greyEq = self.equalize_hist(rotation)
                hists = []
                split = self.split_img(greyEq)
                for img in split:
                    hist = self.get_hog(img)
                    hists.append(hist)
                avgHist = self.getAvgHist(hists)
                glcm = self.get_glcm_features(greyEq)
                trainData.append([avgHist, glcm, 1])
        for image in notPresent:
            rotations = self.get_rotations(image.img)
            for rotation in rotations:
                greyEq = self.equalize_hist(rotation)
                hists = []
                split = self.split_img(greyEq)
                for img in split:
                    hist = self.get_hog(img)
                    hists.append(hist)
                avgHist = self.getAvgHist(hists)
                glcm = self.get_glcm_features(greyEq)
                trainData.append([avgHist, glcm, 0])
        
        avgHists = []
        GLCMs = []
        y = []
        for i in range(len(trainData)):
            avgHists.append(trainData[i][0])
            GLCMs.append(trainData[i][1])
            y.append(trainData[i][2])
        avgHistsNormal, self.HOGmin, self.HOGmax = self.normalizeArr(avgHists)
        GLCMsNormal, self.GLCMmin, self.GLCMmax = self.normalizeArr(GLCMs)
        X = []
        avgHistsNormal = np.array(avgHistsNormal)
        GLCMsNormal = np.array(GLCMsNormal)
        for i in range(len(avgHistsNormal)):
            X.append(np.concatenate((avgHistsNormal[i], GLCMsNormal[i])))
        # X = avgHistsNormal

        self.clf = SVC(C=1000, gamma=.1, kernel='poly', class_weight='balanced')
        if(X == [] or y == []): return
        self.pca = PCA(6)
        X_PCA = self.pca.fit_transform(X)
        X_PCA = np.array(X_PCA)
        self.clf.fit(X_PCA, y)
        getCLFTime = datetime.now()
        print('make CLF time: ' + str(getCLFTime - start))
        dump(self.clf, 'clf.pk1')
        dump(self.pca, 'pca.pk1')

    def read_clf_and_pca(self):
        self.clf = load('clf.pk1')
        self.pca = load('pca.pk1')

    def kfold_present(self, presImages, notPresImages):
        colors = []
        notPresArr = []
        #FILENAME BINPRED AVG
        results_p = []
        results_np = []
        # presImages = self.focusImages(presImages)
        # notPresImages = self.focusImages(notPresImages)
        i = 0
        while(i < len(presImages)):
            color = presImages[i].filename.split()[0]
            colorArr = []
            while(i < len(presImages) and presImages[i].filename.split()[0] == color):
                colorArr.append(presImages[i])
                i+=1
            colors.append(colorArr)

        i = 0
        while(i < len(notPresImages)):
            j = 0
            npImageSet = []
            while(j<= 6):
                npImageSet.append(notPresImages[i])
                j+=1
                i+=1
            notPresArr.append(npImageSet)
        notPresArr.append(notPresArr[len(notPresArr)-1])
        
        i=0
        while(i < len(colors) and i < len(notPresArr)):
            testPres = colors[i]
            trainP2D = colors[:i] + colors[i+1:]
            trainPres = [item for sublist in trainP2D for item in sublist]
            testNotPres = notPresArr[i]
            trainNP2D = notPresArr[:i] + notPresArr[i+1:]
            trainNotPres = [item for sublist in trainNP2D for item in sublist]
            print(testPres)
            print(testNotPres)
            time = datetime.now()
            print('making clfs '+ str(time))
            self.make_clf_and_pca(trainPres, trainNotPres)
            # self.clf = self.readCLFandPCA()
            print("predicting")
            self.predict_presence(testPres)
            self.predict_presence(testNotPres)
            i+=1
        path = os.path.join(self.package_dir, 'GLCM.txt')
        with open(path, 'w') as f:
            f.write('dists\n')
            for key in self.dists:
                f.write('%s %s\n' % (key, self.dists[key]))      
        
    def normalizeArr(self, values, min = None, max = None):
        vals = np.array(values, copy=True, dtype=np.float)
        returnMins = False
        if(min is None or max is None):
            max = np.amax(vals)
            min = np.amin(vals)
            returnMins = True
        vals = (vals - min)/(max - min)
        if(returnMins == True): return vals, min, max
        else: return vals

    def getAvgHist(self, hists):
        sum = np.full(len(hists[0]), 0)
        for hist in hists:
            sum = np.add(sum, hist)
        divisor = np.full(len(hists[0]), len(hists))
        average = np.divide(sum, divisor)
        return average

    def cross_validate(self, present, not_present):
        start = datetime.now()
        trainData = []
        # for image in present:
        #     print(image.filename)
        #     rotations = self.getRotations(image.img)
        #     for rotation in rotations:
        #         greyEq = self.equalize_hist(rotation)
        #         hists = []
        #         split = self.splitImg(greyEq)
        #         for img in split:
        #             hist = self.getHOG(img)
        #             hists.append(hist)
        #         avgHist = self.getAvgHist(hists)
        #         glcm = self.get_GLCM_features(greyEq)
        #         trainData.append([avgHist, glcm, 1])
        #     cv2.imshow(image.filename, greyEq)
        #     cv2.waitKey(0)
        for image in not_present:
            print(image.filename)
            rotations = self.get_rotations(image.img)
            for rotation in rotations:
                greyEq = self.equalize_hist(rotation)
                hists = []
                split = self.split_img(greyEq)
                for img in split:
                    hist = self.get_hog(img)
                    hists.append(hist)
                avgHist = self.getAvgHist(hists)
                glcm = self.get_glcm_features(greyEq)
                trainData.append([avgHist, glcm, 0])
            cv2.imshow(image.filename, greyEq)
            cv2.waitKey(0)

        
        avgHists = []
        GLCMs = []
        y = []
        for i in range(len(trainData)):
            avgHists.append(trainData[i][0])
            GLCMs.append(trainData[i][1])
            y.append(trainData[i][2])
        avgHistsNormal, self.HOGmin, self.HOGmax = self.normalizeArr(avgHists)
        GLCMsNormal, self.GLCMmin, self.GLCMmax = self.normalizeArr(GLCMs)
        X = []
        avgHistsNormal = np.array(avgHistsNormal)
        GLCMsNormal = np.array(GLCMsNormal)
        for i in range(len(avgHistsNormal)):
            X.append(np.concatenate((avgHistsNormal[i], GLCMsNormal[i])))
        if(X == [] or y == []): return
        self.pca = PCA(6)
        X_PCA = self.pca.fit_transform(X)
        X_PCA = np.array(X_PCA)
        clf = SVC(C=1000, gamma=.1, kernel='poly', class_weight='balanced')
        print('validating')
        scores = cross_validate(clf, X, y, scoring='accuracy', cv=5, verbose=3, n_jobs=-1)
        print('done in {}s'.format(scores['fit_time']))
        print(scores['test_score'])
        # self.clf.fit(X_PCA, y)
        getCLFTime = datetime.now()
        print('make CLF time: ' + str(getCLFTime - start))
        dump(self.clf, 'clf.pk1')
        dump(self.pca, 'pca.pk1')

    def __init__(self):
        # np.set_printoptions(threshold=sys.maxsize)
        start = datetime.now()
        self.dists = {}
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        train_pres_imgs = self.get_images(os.path.join(self.package_dir,'images/SVM_training_present'))
        train_not_pres_imgs = self.get_images(os.path.join(self.package_dir, 'images/SVM_training_not_present'))

        self.cross_validate(train_pres_imgs, train_not_pres_imgs)
        # self.kfold_present(trainPresImgs,trainNotPresImgs)
        print('kfold time: ', datetime.now() - start)
        return
        test_pres_imgs = self.get_images(os.path.join(self.package_dir,'images\SVM_test_present'))
        test_not_pres_imgs = self.get_images(os.path.join(self.package_dir,'images\SVM_test_not_present'))

        self.writeHOGsToFile(test_pres_imgs)
        return
        self.getSVM_CLF(test_pres_imgs, test_not_pres_imgs)
        return

        newCLF = True
        if newCLF:
            self.clf = self.make_clf_and_pca(train_pres_imgs, train_not_pres_imgs)
        else: 
            clf = self.read_clf_and_pca()
        print("present")
        for present_img in test_pres_imgs:
            present_prediction = self.predict_presence(present_img)
            print(present_img.filename + ' present prediction: ' + str(present_prediction))
        print("not present")
        for not_present_img in test_not_pres_imgs:
            not_present_prediction = self.predict_presence(not_present_img)
            print(not_present_img.filename + ' not present prediction: ' + str(not_present_prediction))
    



if __name__ == '__main__':
    fd = FabricDetector()