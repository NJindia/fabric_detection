import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.feature import hog, greycoprops, greycomatrix
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.decomposition import PCA
from joblib import load, dump
import mahotas as mt
from datetime import datetime


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
            pred = 0
            hists = []
            roi = self.getROI(image.img)
            split = self.splitImg(roi)            
            for img in split:
                hist = self.getHOG(img)
                hists.append(hist)
            avgHist = self.getAvgHist(hists)
            
            grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # gcm = greycomatrix(grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=256)
            # gcp = greycoprops(gcm, prop='dissimilarity')[0]
            haralick = self.extractHaralickFeats(grey)

            # gcpNormalized = self.normalizeArr(gcp, min=self.GLCMmin, max = self.GLCMmax)
            haralickNormalized = self.normalizeArr(haralick, min=self.haralickMin, max = self.haralickMax)
            avgHistNormalized = self.normalizeArr(avgHist, min=self.HOGmin, max=self.HOGmax)
            
            predArr = np.concatenate((avgHistNormalized, haralickNormalized))
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

    def makeCLFandPCA(self, present, notPresent):
        start = datetime.now()
        trainData = []
        for image in present:
            rotations = self.getRotations(image.img)
            for rotation in rotations:
                hists = []
                split = self.splitImg(rotation)
                for img in split:
                    hist = self.getHOG(img)
                    hists.append(hist)
                avgHist = self.getAvgHist(hists)
                grey = cv2.cvtColor(rotation, cv2.COLOR_BGR2GRAY)
                # gcm = greycomatrix(grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                #     levels=256)
                # glcm = greycoprops(gcm, prop='dissimilarity')[0]
                haralick = self.extractHaralickFeats(grey)

                trainData.append([avgHist, haralick, 1])
        for image in notPresent:
            rotations = self.getRotations(image.img)
            for rotation in rotations:
                hists = []
                split = self.splitImg(rotation)
                for img in split:
                    hist = self.getHOG(img)
                    hists.append(hist)
                avgHist = self.getAvgHist(hists)
                grey = cv2.cvtColor(rotation, cv2.COLOR_BGR2GRAY)
                # gcm = greycomatrix(grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                #     levels=256)
                # glcm = greycoprops(gcm, prop='dissimilarity')[0]
                haralick = self.extractHaralickFeats(grey)

                trainData.append([avgHist, haralick, 0])
        avgHists = []
        haralickFeats = []
        y = []
        for i in range(len(trainData)):
            avgHists.append(trainData[i][0])
            haralickFeats.append(trainData[i][1])
            y.append(trainData[i][2])
        avgHistsNormal, self.HOGmin, self.HOGmax = self.normalizeArr(avgHists)
        # GLCMsNormal, self.GLCMmin, self.GLCMmax = self.normalizeArr(GLCMs)
        haralickFeatsNormal, self.haralickMin, self.haralickMax = self.normalizeArr(haralickFeats)
        X = []
        avgHistsNormal = np.array(avgHistsNormal)
        haralickFeatsNormal = np.array(haralickFeatsNormal)
        for i in range(len(avgHistsNormal)):
            X.append(np.concatenate((avgHistsNormal[i], haralickFeatsNormal[i])))
        # X = avgHistsNormal
        X = np.array(X)
        # self.eigen(X)

        self.clf = svm.SVC(C=.8)
        if(X == [] or y == []): return
        self.pca = PCA(n_components=6)
        X_PCA = self.pca.fit_transform(X)
        X_PCA = np.array(X_PCA)
        self.clf.fit(X_PCA, y)
        getCLFTime = datetime.now()
        print('make CLF time: ' + str(getCLFTime - start))
        dump(self.clf, 'clf.pk1')
        dump(self.pca, 'pca.pk1')

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
            color = presImages[i].fileName.split()[0]
            colorArr = []
            while(i < len(presImages) and presImages[i].fileName.split()[0] == color):
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
            # self.clf = self.readCLFandPCA()
            print("predicting")
            self.predictPresence(testPres)
            self.predictPresence(testNotPres)
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

    def __init__(self):
        # np.set_printoptions(threshold=sys.maxsize)
        start = datetime.now()
        self.dists = {}
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        trainPresImgs = self.getImages(os.path.join(self.package_dir,'images/SVM_training_present'))
        trainNotPresImgs = self.getImages(os.path.join(self.package_dir, 'images/SVM_training_not_present'))

        # self.focusImages(trainPresImgs)
        # self.focusImages(trainNotPresImgs)

        self.kfold_present(trainPresImgs,trainNotPresImgs)
        print('kfold time: ', datetime.now() - start)
        return
        testPresImgs = self.getImages(os.path.join(self.package_dir,'images\SVM_test_present'))
        testNotPresImgs = self.getImages(os.path.join(self.package_dir,'images\SVM_test_not_present'))

        #1 = high acutance, 2 = low acutance

        self.writeHOGsToFile(testPresImgs)
        return
        # testPresImgs = self.focusImages(testPresImgs)
        # testNotPresImgs = self.focusImages(testNotPresImgs)
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