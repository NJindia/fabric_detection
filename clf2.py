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
from skimage import color, data, restoration
from numpy.fft import fft2, ifft2

class Image:
    def __init__(self, img=None, fileName=None, path=None, clf=None):
        self.img = img
        self.fileName = fileName
        self.path = path
        self.clf = clf

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


    def predictPresence(self, image):
        clf = 0
        sum = 0
        roi = self.getROI(image.img)
        split = self.splitImg(roi)
        p = 0
        for im in split:
            hist = self.getHOG(im)
            hist = hist.reshape(1, -1)
            if(image.clf == 1):
                p = self.clf1.predict(hist)
            elif(image.clf == 2):
                p = self.clf2.predict(hist)
            if(p==1):
                sum += 1
        pred = sum/len(split)
        if(pred > 0.5):
            return 1, pred
        else:
            return 0, pred

    def getSVM_CLF(self, present, notPresent):
        X = []
        y = []
        for image in present:
            # print(image.fileName)
            rotations = self.getRotations(image.img)
            for rotation in rotations:
                split = self.splitImg(rotation)
                for img in split:
                    X.append(self.getHOG(img))
                    y.append(1)
        for image in notPresent:
            # print(image.fileName)
            rotations = self.getRotations(image.img)
            for rotation in rotations:
                split = self.splitImg(rotation)
                for img in split:
                    X.append(self.getHOG(img))
                    y.append(0)
        clf = svm.SVC()
        if(X == [] or y == []): return
        clf.fit(X, y)
        return clf

    def makeCLF(self, present, notPresent):
        # start = datetime.now()
        # # print('start')
        # # present = self.getImages(os.path.join(self.package_dir,'images/SVM_training_present'))
        # # notPresent = self.getImages(os.path.join(self.package_dir, 'images/SVM_training_not_present'))
        getImgsTime = datetime.now()
        # print('get Images time: ' + str(getImgsTime - start))
        clf = self.getSVM_CLF(present, notPresent)
        getCLFTime = datetime.now()
        print('make CLF time: ' + str(getCLFTime - getImgsTime))
        # dump(clf, 'clf.pk1')
        return clf

    def readCLF(self):
        start = datetime.now()
        clf = load('clf.pk1')
        loadCLFTime = datetime.now()
        print('load CLF time: ' + str(loadCLFTime - start))

        return clf

    def getAcutance(self, image):
        if(type(image) == Image): img = self.getROI(image.img)
        else: img = image
        lbound = 25
        ubound = 150
        edges = cv2.Canny(img,lbound,ubound) 
        acutance = np.mean(edges)
        return acutance

    def getCLFfocusImage(self, images):
        retImages = []
        tempImages = images[:]
        #1 = high acutance, 2 = low acutance
        psf = self.create_gauss_kernel(sigma=5)
        for image in tempImages:
            acutance = self.getAcutance(image)
            if(acutance > self.acutance_thresh):
                image.clf = 1
                image.img = self.shiftHist(image.img)
            else: 
                # image.img = self.shiftHist(image.img)
                image.clf = 2
                image.img = self.wiener(image, psf)
            retImages.append(image)
        return retImages

    def imageNotPresentSubset(self, notPresentImages):
        notPresArr = []
        i = 0
        while(i < len(notPresentImages)):
            j = 0
            npImageSet = []
            while(j<= 6):
                npImageSet.append(notPresentImages[i])
                j+=1
                i+=1
            notPresArr.append(npImageSet)
        notPresArr.append(notPresArr[len(notPresArr)-1]) #hard coded to add a duplicate of last set
        return notPresArr

    def kfold_present(self, presImages, notPresentImages, notPresentImagesFocused):
        colors = []
        #FILENAME BINPRED AVG
        clf1_results_p = []
        clf2_results_p = []
        clf1_results_np = []
        clf2_results_np = []
        i = 0
        while(i < len(presImages)):
            color = presImages[i].fileName.split()[0]
            colorArr = []
            while(i < len(presImages) and presImages[i].fileName.split()[0] == color):
                colorArr.append(presImages[i])
                i+=1
            colors.append(colorArr)
        notPresArr = self.imageNotPresentSubset(notPresentImages)
        notPresArrFocused = self.imageNotPresentSubset(notPresentImagesFocused)
        
        i=0
        while(i < len(colors) and i < len(notPresArr)):
            testPres = colors[i]
            trainP2D = colors[:i] + colors[i+1:]
            trainPres = [item for sublist in trainP2D for item in sublist]
            # trainPres1, trainPres2 = self.filterByAcutance(trainPres)
            trainPres1 = []
            trainPres2 = []
            for image in trainPres:
                if(image.clf == 1):
                    trainPres1.append(image)
                elif(image.clf == 2):
                    trainPres2.append(image)
            testNotPres = notPresArr[i]
            trainNP2D = notPresArr[:i] + notPresArr[i+1:]
            trainNotPres = [item for sublist in trainNP2D for item in sublist]

            trainNP2DF = notPresArrFocused[:i] + notPresArrFocused[i+1:]
            trainNotPresFocused = [item for sublist in trainNP2D for item in sublist]

            print(testPres)
            print('\n')
            print(trainPres1)
            print(trainPres2)
            print('\n')
            print(testNotPres)
            time = datetime.now()
            print('making clf1 '+ str(time))
            self.clf1 = self.makeCLF(trainPres1, trainNotPres)
            time = datetime.now()
            print('making clf2 '+ str(time))
            self.clf2 = self.makeCLF(trainPres2, trainNotPresFocused)
            print("predicting")
            for image in testPres:
                prediction, avg= self.predictPresence(image)
                print(image.clf, image.fileName, prediction, avg)
                if(image.clf == 1):
                    clf1_results_p.append((image.fileName, prediction, avg))
                elif(image.clf == 2):
                    clf2_results_p.append((image.fileName, prediction, avg))
            for image in testNotPres:
                prediction, avg = self.predictPresence(image)
                print(image.clf, image.fileName, prediction, avg)
                if(image.clf == 1):
                    clf1_results_np.append((image.fileName, prediction, avg))
                elif(image.clf == 2):
                    clf2_results_np.append((image.fileName, prediction, avg))
            i+=1
        path = os.path.join(self.package_dir, '2clf.txt')
        with open(path, 'w') as f:
            f.write('clf1 > 5 acutance\n')
            f.write('present\n')
            for prediction in clf1_results_p:
                f.write('%s %s %s\n' % (prediction[0], prediction[1], prediction[2]))
            f.write('not present\n')
            for prediction in clf1_results_np:
                f.write('%s %s %s\n' % (prediction[0], prediction[1], prediction[2]))
            f.write('clf2 < 5 acutance\n')
            f.write('present\n')
            for prediction in clf2_results_p:
                f.write('%s %s %s\n' % (prediction[0], prediction[1], prediction[2]))
            f.write('not present\n')
            for prediction in clf2_results_np:
                f.write('%s %s %s\n' % (prediction[0], prediction[1], prediction[2]))

    # Gaussian kernel generation function
    def create_gauss_kernel(self, kernel_size=3, sigma=1, k=1):
        if sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        X = np.linspace(-k, k, kernel_size)
        Y = np.linspace(-k, k, kernel_size)
        x, y = np.meshgrid(X, Y)
        x0 = 0
        y0 = 0
        gauss = 1/(2*np.pi*sigma**2) * np.exp(- ((x -x0)**2 + (y - y0)**2)/ (2 * sigma**2))
        gaussian = gauss/(np.sum(gauss))
        return gaussian


    def wiener(self, image, psf):
        img = image.img
        og = img
        before = self.getAcutance(img)  
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = color.rgb2gray(img)
        deconvolved, _ = restoration.unsupervised_wiener(img, psf)
        deconvolved = (deconvolved*255).astype(np.uint8)
        print(image.fileName, before, self.getAcutance(deconvolved))

        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
        #                     sharex=True, sharey=True)

        # plt.gray()

        # ax[0].imshow(img)
        # ax[0].axis('off')
        # ax[0].set_title(image.fileName)

        # ax[1].imshow(deconvolved)
        # ax[1].axis('off')
        # ax[1].set_title(str(j))

        # fig.tight_layout()

        # plt.show()
        return deconvolved

    def __init__(self):
        self.acutance_thresh = 5


        start = datetime.now()
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        trainPresImgs = self.getImages(os.path.join(self.package_dir,'images/SVM_training_present'))
        trainNotPresImgs = self.getImages(os.path.join(self.package_dir, 'images/SVM_training_not_present'))

        trainPresImgsFocused = self.getCLFfocusImage(trainPresImgs)
        trainNotPresImgsFocused = self.getCLFfocusImage(trainNotPresImgs)
        for image in trainNotPresImgsFocused:
            print(image.fileName, image.clf)

        self.kfold_present(trainPresImgsFocused,trainNotPresImgs, trainNotPresImgsFocused)
        print('kfold time: ', datetime.now() - start)
        return
        # testPresImgs = self.getImages(os.path.join(self.package_dir,'images\SVM_test_present'))
        # testNotPresImgs = self.getImages(os.path.join(self.package_dir,'images\SVM_test_not_present'))
        # #1 = high acutance, 2 = low acutance
        
        
        
        # trainPres1, trainPres2 = self.filterByAcutance(trainPresImgs)
        # self.wiener(trainPres2)

        # newCLF = True
        # if newCLF:
        #     self.clf1 = self.makeCLF(trainPres1, trainNotPresImgs)
        #     self.clf2 = self.makeCLF(trainPres2, trainNotPresImgs)
        # else: 
        #     clf = self.readCLF()
        # # np.savetxt('avgPresentHist.csv', avgPresentHist, fmt='%s', delimiter=',', header="value")
        # # np.savetxt('avgNotPresentHist.csv', avgNotPresentHist, fmt='%s', delimiter=',', header="value")    
        # print("present")
        # for presImg in testPresImgs:
        #     presPrediction = self.predictPresence(presImg)
        #     print(presImg.fileName + ' present prediction: ' + str(presPrediction))
        # print("not present")
        # for notPresImg in testNotPresImgs:
        #     notPresPrediction = self.predictPresence(notPresImg)
        #     print(notPresImg.fileName + ' not present prediction: ' + str(notPresPrediction))
    



if __name__ == '__main__':
    fd = FabricDetector()