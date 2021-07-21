import numpy as np
import os
import matplotlib as plt
import cv2
from joblib import load, dump
from sklearn import svm
from sklearn.decomposition import PCA
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern
import datetime
import mahotas as mt



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

    def increaseContrast(self, image):
        lab = cv2.cvtColor(image.img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=1000.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.imshow(image.fileName, final)
        cv2.waitKey(0)

        return final

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

    def getDist(self, images):
        X = []
        for image in images:
            sum = 0
            pred = 0
            hists = []
            roi = self.getROI(image.img)
            #TODO SPLIT?
            grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean, std = self.extractHaralickFeats(grey)
            X.append(mean) #HERE
        X_PCA = self.pca.transform(X)
        dist = self.clf.decision_function(X_PCA)
        return dist

    def makeCLFandPCA(self, present, notPresent):
        start = datetime.now()
        X_means = []
        X_stds = []
        y = []
        for image in present:
            rotations = self.getRotations(image.img)
            for rotation in rotations:
                grey = cv2.cvtColor(rotation, cv2.COLOR_BGR2GRAY)
                # gcm = greycomatrix(grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                #     levels=256)
                # glcm = greycoprops(gcm, prop='dissimilarity')[0]
                mean, std = self.extractHaralickFeats(grey)
                X_means.append(mean)
                X_stds.append(std)
                y.append(1)
        for image in notPresent:
            rotations = self.getRotations(image.img)
            for rotation in rotations:
                grey = cv2.cvtColor(rotation, cv2.COLOR_BGR2GRAY)
                # gcm = greycomatrix(grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                #     levels=256)
                # glcm = greycoprops(gcm, prop='dissimilarity')[0]
                mean, std = self.extractHaralickFeats(grey)
                X_means.append(mean)
                X_stds.append(std)
                y.append(0)
        meansNormal, self.meanMin, self.meanMax = self.normalizeArr(X_means)
        stdsNormal, self.stdMin, self.stdMax = self.normalizeArr(X_stds)
        self.clf = svm.SVC()
        self.pca = PCA(5)
        X_PCA = self.pca.fit_transform(meansNormal) #HERE
        X_PCA = np.array(X_PCA)
        self.clf.fit(X_PCA, y)
        getCLFTime = datetime.now()
        print('make CLF time: ' + str(getCLFTime - start))
        dump(self.clf, 'clf.pk1')
        dump(self.pca, 'pca.pk1')
   
    def getRotations(self, img):
        rotations = []
        for i in range(0, 360, 15):
            image_center = tuple(np.array(img.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, i, 1.0)
            rotation = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            roi = self.getROI(rotation)
            rotations.append(roi)
        return rotations

    def extractHaralickFeats(self, grey):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(grey)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        ht_std = textures.std(axis=0)
        return ht_mean, ht_std

    def texture_detect(self, images):
        radius = 1
        n_point = 1
        train_hist = []
        for image in images:
            grey = cv2.cvtColor(image.img, cv2.COLOR_BGR2GRAY)
            #Use the LBP method to extract the texture features of the image.
            lbp=local_binary_pattern(grey,n_point,radius,'default')
            cv2.imshow(image.fileName, lbp)
            cv2.waitKey(0)
            #Statistic image histogram
            #hist size:256
            max_bins = 36
            hist = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
            print(image.fileName, hist)
            train_hist.append(hist)
        return train_hist
        
    def __init__(self):
        package_dir = os.path.dirname(os.path.abspath(__file__))
        pres = self.getImages(os.path.join(package_dir,'images/SVM_training_present'))
        notPres = self.getImages(os.path.join(package_dir,'images/SVM_training_not_present'))
        testPres = self.getImages(os.path.join(package_dir,'images/SVM_test_present'))
        testNotPres = self.getImages(os.path.join(package_dir,'images/SVM_test_not_present'))
        self.makeCLFandPCA(pres, notPres)
        
        
        # self.texture_detect(pres)
        # self.texture_detect(notPres)
        # for image in notPres:
        #     # self.increaseContrast(image)
        #     rotations = self.getRotations(image.img)
        #     for rotation in rotations:
        #         grey = cv2.cvtColor(rotation, cv2.COLOR_BGR2GRAY)
        #         # gcm = greycomatrix(grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
        #         #     levels=256)
        #         # glcm = greycoprops(gcm, prop='ASM')[0]
        #         mean, std = self.extractHaralickFeats(grey)
        #         print(image.fileName)
        #         print('mean', mean)
        #         print('std', std)
        # print('not pres')
        # for image in notPres:
        #     # self.increaseContrast(image)
        #     rotations = self.getRotations(image.img)
        #     for rotation in rotations:
        #         grey = cv2.cvtColor(rotation, cv2.COLOR_BGR2GRAY)
        #         # gcm = greycomatrix(grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
        #         #     levels=256)
        #         # glcm = greycoprops(gcm, prop='ASM')[0]
        #         mean, std = self.extractHaralickFeats(grey)
        #         print(image.fileName)
        #         print('mean', mean)
        #         print('std', std)
if __name__ == '__main__':
    fd = FabricDetector()