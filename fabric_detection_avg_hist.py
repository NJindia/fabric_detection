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
from sklearn.model_selection import GridSearchCV
from joblib import load, dump
import mahotas as mt
from datetime import datetime
import plot_equalize

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
                imgs.append(image)
        return imgs

    def getHOG(self, img):
        hist = hog(img, orientations=36, pixels_per_cell=(100, 100), cells_per_block=(1, 1))
        return hist

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

    def getRotations(self, img):
        rotations = []
        for i in range(0, 360, 15):
            image_center = tuple(np.array(img.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, i, 1.0)
            rotation = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            roi = self.getROI(rotation)
            rotations.append(roi)
        return rotations
 
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
        
    def predictPresence(self, images):
        predictions = []
        for image in images:
            # hists = []
            roi = self.getROI(image.img)
            # split = self.splitImg(roi)            
            # for img in split:
            #     hist = self.getHOG(img)
            #     hists.append(hist)
            # avgHist = self.getAvgHist(hists)
            index = image.fileName + '0' #DEBUG
            avgHist = self.features[index][0] #DEBUG
            glcm = self.features[index][1]
            haralick = self.features[index][2]
            
            # grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # gcm = greycomatrix(grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=256)
            # gcp = greycoprops(gcm, prop='dissimilarity')[0]
            # haralick = self.extractHaralickFeats(grey)

            # gcpNormalized = self.normalizeArr(gcp, min=self.GLCMmin, max = self.GLCMmax)
            haralickNormalized = self.normalizeArr(haralick, min=self.haralickMin, max = self.haralickMax)
            avgHistNormalized = self.normalizeArr(avgHist, min=self.HOGmin, max=self.HOGmax)
            GLCM_normalized = self.normalizeArr(glcm, min=self.GLCMmin, max=self.GLCMmax)
            predArr = np.concatenate((avgHistNormalized, GLCM_normalized)) #MODIFY
            # predArr = avgHistNormalized
            predArr = predArr.reshape(1, -1)

            PCA = self.pca.transform(predArr)
            pred = self.clf.predict(PCA)
            predictions.append([image.fileName, pred])

            #DEBUG
            dist = self.clf.decision_function(PCA)
            self.dists[image.fileName] = dist
            print(image.fileName, dist)
        return predictions

    def scoreDists(self):
        currColor = ''
        colorDict = {}
        distDict = {}
        colorArr = []
        not_present_arr = []
        for key in self.dists:   
            if(key.split()[0] == 'not_present'):
                not_present_arr.append(self.dists[key])
                continue
            newColor = key.split()[0]
            if(currColor != newColor): 
                if(currColor == ''):
                    currColor = newColor
                else:
                    colorArr = np.array(colorArr)
                    colorDict['mean'] = np.mean(colorArr)
                    colorDict['std'] = np.std(colorArr)
                    colorDict['min'] = np.min(colorArr)
                    colorDict['max'] = np.max(colorArr)
                    colorDict['n'] = len(colorArr)
                    distDict[currColor] = colorDict
                    currColor = newColor
                    colorDict = {}
                    colorArr = []
            colorArr.append(self.dists[key])
        colorArr = np.array(colorArr)
        colorDict['mean'] = np.mean(colorArr)
        colorDict['std'] = np.std(colorArr)
        colorDict['min'] = np.min(colorArr)
        colorDict['max'] = np.max(colorArr)
        colorDict['n'] = len(colorArr)
        distDict[currColor] = colorDict
        colorDict = {}
        not_present_arr = np.array(not_present_arr)
        colorDict['mean'] = np.mean(not_present_arr)
        colorDict['std'] = np.std(not_present_arr)
        colorDict['min'] = np.min(not_present_arr)
        colorDict['max'] = np.max(not_present_arr)
        colorDict['n'] = len(not_present_arr)
        distDict['not_present'] = colorDict

        return distDict

    def pickleFeatures(self, images):
        feats = {}
        for image in images:
            print(image.fileName)
            rotations = self.getRotations(image.img)
            for i in range(len(rotations)):
                rotation = rotations[i]
                grey = cv2.cvtColor(rotation, cv2.COLOR_BGR2GRAY)
                rotationEq = exposure.equalize_adapthist(grey, clip_limit=0.5)
                hists = []
                split = self.splitImg(rotationEq)
                for img in split:
                    hist = self.getHOG(img)
                    hists.append(hist)
                avgHist = self.getAvgHist(hists)
                # try:
                #     rotation.shape[2]
                #     grey = cv2.cvtColor(rotationEq, cv2.COLOR_BGR2GRAY)
                # except: grey = rotationEq
                grey2 = np.uint8(255*rotationEq)
                gcm = greycomatrix(grey2, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256)
                glcm = greycoprops(gcm, prop='dissimilarity')[0]
                haralick = self.extractHaralickFeats(grey2)
                index = str(image.fileName) + str(i)
                feats[index] = [avgHist, glcm, haralick]
        self.features = feats
        dump(self.features, 'featuresEq.pickle')

    def makeCLFandPCA(self, present, notPresent):
        t0 = time()
        trainData = []
        for image in present:
            for i in range(0, 360, 15):
                index = str(image.fileName) + str(int(i/15))
                avgHist = self.features[index][0]
                glcm = self.features[index][1]
                haralick = self.features[index][2]
                trainData.append([avgHist, haralick, glcm, 1])
        for image in notPresent:
            for i in range(0, 360, 15):
                index = str(image.fileName) + str(int(i/15))
                avgHist = self.features[index][0]
                glcm = self.features[index][1]
                haralick = self.features[index][2]
                trainData.append([avgHist, haralick, glcm, 0])
        avgHists = []
        haralickFeats = []
        GLCMs = []
        y = []
        for i in range(len(trainData)):
            avgHists.append(trainData[i][0])
            haralickFeats.append(trainData[i][1])
            GLCMs.append(trainData[i][2])
            y.append(trainData[i][3])
        avgHistsNormal, self.HOGmin, self.HOGmax = self.normalizeArr(avgHists)
        GLCMsNormal, self.GLCMmin, self.GLCMmax = self.normalizeArr(GLCMs)
        haralickFeatsNormal, self.haralickMin, self.haralickMax = self.normalizeArr(haralickFeats)
        X = []
        avgHistsNormal = np.array(avgHistsNormal)
        haralickFeatsNormal = np.array(haralickFeatsNormal)
        GLCMsNormal = np.array(GLCMsNormal)
        for i in range(len(avgHistsNormal)):
            X.append(np.concatenate((avgHistsNormal[i], GLCMsNormal[i])))
        # X = avgHistsNormal
        X = np.array(X)
        # self.eigen(X)
        if(X == [] or y == []): return
        self.pca = PCA(n_components=6)
        X_PCA = self.pca.fit_transform(X)
        X_PCA = np.array(X_PCA)
        # self.gridSearch(X, y)
        # return
        self.clf = SVC(kernel='poly', C=1000.0, gamma=0.1, class_weight='balanced') #TODO MODIFY
        self.clf.fit(X_PCA, y)
        print('make CLF time: ' + str(time() - t0))
        dump(self.clf, 'clf.pk1')
        dump(self.pca, 'pca.pk1')


    def gridSearch(self, X_train, y_train):
        t0 = time()
        print('GridSearch Start{}'.format(datetime.now()))
        #Create a dictionary of possible parameters
        param_grid = {'C': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                        'kernel':['rbf', 'poly'],
                        'gamma':[1e-1, 1e-2, 1e-3, 1e-4]
                    }
        lin_grid = {'C': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5]}
        #Create the GridSearchCV object
        # grid_clf = GridSearchCV(LinearSVC(random_state=0, class_weight='balanced'), param_grid, n_jobs=1, verbose=3)
        grid_clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, n_jobs=-1, cv=10, verbose=2)
        #Fit the data with the best possible parameters
        grid_clf = grid_clf.fit(X_train, y_train)
        #Print the best estimator with it's parameters
        s1 = "done in %0.3fs" % (time() - t0)
        s2 = 'best estimators: '
        s3 = grid_clf.best_params_
        s4 = grid_clf.best_score_
        s5 = grid_clf.best_index_
        print(s1)
        print(s2)
        print(s3)
        print(s4)
        print(s5)
        for i in range(len(grid_clf.cv_results_['params'])):
            print('{}: {} {}'.format(i, grid_clf.cv_results_['params'][i], grid_clf.cv_results_['mean_test_score'][i]))

    def extractHaralickFeats(self, img):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(img)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean

    # @X_train IS 2D
    def eigen(self, X_train):
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        cov_mat = np.cov(X_train_std.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        # calculate cumulative sum of explained variances
        tot = sum(eigen_vals)
        var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)

        # plot explained variances
        plt.bar(range(1,X_train.shape[1]+1), var_exp, alpha=0.5,
                align='center', label='individual explained variance')
        plt.step(range(1,X_train.shape[1]+1), cum_var_exp, where='mid',
                label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.show()

    def readCLFandPCA(self):
        self.clf = load('clf.pk1')
        self.pca = load('pca.pk1')

    def kfold(self, pres, notPres):
        colors = []
        notPresArr = []
        i = 0
        while(i < len(pres)):
            currColor = pres[i].fileName.split()[0]
            colorArr = []
            while(i < len(pres) and pres[i].fileName.split()[0] == currColor):
                colorArr.append(pres[i])
                i+=1
            colors.append(colorArr)

        i = 0
        while(i < len(notPres)):
            j = 0
            npImageSet = []
            while(j<= 6 and i < len(notPres)):
                npImageSet.append(notPres[i])
                j+=1
                i+=1
            notPresArr.append(npImageSet)
        notPresArr.append(notPresArr[len(notPresArr)-1])
        
        i=0
        while(i < len(colors) and i < len(notPresArr)):
            testPres = colors[i]
            # trainP2D = colors[:i] + colors[i+1:]
            trainP2D = colors #TODO DELETE
            trainPres = [item for sublist in trainP2D for item in sublist]
            testNotPres = notPresArr[i]
            trainNP2D = notPresArr[:i] + notPresArr[i+1:]
            trainNotPres = [item for sublist in trainNP2D for item in sublist]
            print(testPres)
            print(testNotPres)
            time = datetime.now()
            print('making clfs '+ str(time))
            self.makeCLFandPCA(trainPres, trainNotPres)
            # return #TODO DELETE
            # self.clf = self.readCLFandPCA()
            print("predicting")
            self.predictPresence(testPres)
            self.predictPresence(testNotPres)
            i+=1
        scoredDists = self.scoreDists()
        for elem in scoredDists: print(elem, scoredDists[elem])
        path = os.path.join(self.package_dir, 'CTest.txt')
        with open(path, 'a') as f:
            f.write('C = {} RBF\n'.format(self.C))
            for elem in scoredDists: f.write(str(elem) + str(scoredDists[elem]) + '\n')    
        
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

    def __init__(self):
        # np.set_printoptions(threshold=sys.maxsize)
        start = datetime.now()
        self.dists = {}
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        trainPresImgs = self.getImages(os.path.join(self.package_dir,'images/SVM_training_present'))
        trainNotPresImgs = self.getImages(os.path.join(self.package_dir, 'images/SVM_training_not_present'))
        self.features = load('featuresEq.pickle')
        # for image in trainNotPresImgs:
        #     roi = self.getROI(image.img)
        #     grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #     plot_equalize.main(grey)
        # self.pickleFeatures(np.concatenate((trainPresImgs, trainNotPresImgs)))
        # return
        # for C in [5e3, 1e4, 5e4, 1e5, 5e5, 1e6]:
        #     self.C = C
        #     print(C)
        self.kfold(trainPresImgs,trainNotPresImgs)
        print('kfold time: ', datetime.now() - start)
        return
        testPresImgs = self.getImages(os.path.join(self.package_dir,'images\SVM_test_present'))
        testNotPresImgs = self.getImages(os.path.join(self.package_dir,'images\SVM_test_not_present'))

        #1 = high acutance, 2 = low acutance

        self.writeHOGsToFile(testPresImgs)
        return
        self.getSVM_CLF(testPresImgs, testNotPresImgs)

        return
        newCLF = True
        if newCLF:
            self.clf = self.makeCLF(trainPresImgs, trainNotPresImgs)
            # self.clf1 = self.makeCLF(trainPres1, trainNotPresImgs)
            # self.clf2 = self.makeCLF(trainPres2, trainNotPresImgs)
        else: 
            clf = self.readCLF()
        # np.savetxt('avgPresentHist.csv', avgPresentHist, fmt='%s', delimiter=',', header="value")
        # np.savetxt('avgNotPresentHist.csv', avgNotPresentHist, fmt='%s', delimiter=',', header="value")    
        print("present")
        for presImg in testPresImgs:
            presPrediction = self.predictPresence(presImg)
            print(presImg.fileName + ' present prediction: ' + str(presPrediction))
        print("not present")
        for notPresImg in testNotPresImgs:
            notPresPrediction = self.predictPresence(notPresImg)
            print(notPresImg.fileName + ' not present prediction: ' + str(notPresPrediction))
    



if __name__ == '__main__':
    fd = FabricDetector()