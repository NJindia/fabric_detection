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
from scipy.signal import gaussian

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
        acutance = self.getAcutance(image)
        sum = 0
        roi = self.getROI(image.img)
        split = self.splitImg(roi)
        for im in split:
            hist = self.getHOG(im)
            hist = hist.reshape(1, -1)
            if(acutance < self.acutance_thresh):
                p = self.clf2.predict(hist)
                clf = 2 #DEBUG
            else:
                p = self.clf1.predict(hist)
                clf = 1 #DEBUG
            if(p==1):
                sum += 1
        pred = sum/len(split)
        if(pred > 0.5):
            return 1, pred, clf #DEBUG
        else:
            return 0, pred, clf #DEBUG

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
        dump(clf, 'clf.pk1')
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


    def filterByAcutance(self, images):
        self.acutance_thresh = 5
        #1 = high acutance, 2 = low acutance
        train1 = []
        train2 = []
        for image in images:
            acutance = self.getAcutance(image)
            if(acutance > self.acutance_thresh):
                train1.append(image)
            else: 
                train2.append(image)
        return train1, train2

    def kfold_present(self, presImages, notPresentImages):
        colors = []
        notPresArr = []
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
        
        i = 0
        while(i < len(notPresentImages)):
            j = 0
            npImageSet = []
            while(j<= 6):
                npImageSet.append(notPresentImages[i])
                j+=1
                i+=1
            notPresArr.append(npImageSet)
        notPresArr.append(notPresArr[len(notPresArr)-1])
        
        i=0
        while(i < len(colors) and i < len(notPresArr)):
            testPres = colors[i]
            trainP2D = colors[:i] + colors[i+1:]
            trainPres = [item for sublist in trainP2D for item in sublist]
            trainPres1, trainPres2 = self.filterByAcutance(trainPres)
            testNotPres = notPresArr[i]
            trainNP2D = notPresArr[:i] + notPresArr[i+1:]
            trainNotPres = [item for sublist in trainNP2D for item in sublist]
            print(testPres)
            print('\n')
            print(testNotPres)
            print('\n')
            
            print('making clfs')
            self.clf1 = self.makeCLF(trainPres1, trainNotPres)
            self.clf2 = self.makeCLF(trainPres2, trainNotPres)
            print("predicting")
            for image in testPres:
                prediction, avg, clf = self.predictPresence(image)
                print(clf, image.fileName, prediction, avg)
                if(clf == 1):
                    clf1_results_p.append((image.fileName, prediction, avg))
                elif(clf == 2):
                    clf2_results_p.append((image.fileName, prediction, avg))
            for image in testNotPres:
                prediction, avg, clf = self.predictPresence(image)
                print(clf, image.fileName, prediction, avg)
                if(clf == 1):
                    clf1_results_np.append((image.fileName, prediction, avg))
                elif(clf == 2):
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

    def wiener_filter(self, img, kernel, K):
        kernel /= np.sum(kernel)
        dummy = np.copy(img)
        dummy = fft2(dummy)
        kernel = fft2(kernel, s = img.shape)
        kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
        dummy = dummy * kernel
        dummy = np.abs(ifft2(dummy))
        return dummy

    def gaussian_kernel(self, kernel_size = 3):
        h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
        h = np.dot(h, h.transpose())
        h /= np.sum(h)
        return h

    def wiener(self, images):
        print('\nwiener\n')
        for image in images:
            img = image.img
            og = img
            before = self.getAcutance(img)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = color.rgb2gray(img)
            for i in range(1,4):        
                psf = np.ones((i, i)) / (i*i)
                # img = conv2(img, psf, 'same')
                # img += 0.1 * img.std() * np.random.standard_normal(img.shape)

                deconvolved, _ = restoration.unsupervised_wiener(img, psf)
                print(before, self.getAcutance((deconvolved*255).astype(np.uint8)))

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                                    sharex=True, sharey=True)

                plt.gray()

                ax[0].imshow(img)
                ax[0].axis('off')
                ax[0].set_title('Data')

                ax[1].imshow(deconvolved)
                ax[1].axis('off')
                ax[1].set_title(str(i))

                fig.tight_layout()

                plt.show()

    def __init__(self):
        start = datetime.now()
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        trainPresImgs = self.getImages(os.path.join(self.package_dir,'images/SVM_training_present'))
        trainNotPresImgs = self.getImages(os.path.join(self.package_dir, 'images/SVM_training_not_present'))
        self.kfold_present(trainPresImgs,trainNotPresImgs)
        print('kfold time: ', datetime.now() - start)
        return
        testPresImgs = self.getImages(os.path.join(self.package_dir,'images\SVM_test_present'))
        testNotPresImgs = self.getImages(os.path.join(self.package_dir,'images\SVM_test_not_present'))
        #1 = high acutance, 2 = low acutance
        
        
        
        trainPres1, trainPres2 = self.filterByAcutance(trainPresImgs)
        # self.wiener(trainPres2)

        newCLF = True
        if newCLF:
            self.clf1 = self.makeCLF(trainPres1, trainNotPresImgs)
            self.clf2 = self.makeCLF(trainPres2, trainNotPresImgs)
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