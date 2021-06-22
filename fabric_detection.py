import cv2
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import maximum
from skimage.feature import hog
from skimage import data, exposure
from sklearn import svm
from sklearn.decomposition import PCA
from joblib import load, dump
from datetime import datetime
from skimage import color, data, restoration
from numpy.fft import fft2, ifft2
import xlsxwriter


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
            sum = 0
            pred = 0
            hists = []
            # rotations = self.getRotations(image.img)
            # hists = []
            # for rotation in rotations:
            #     split = self.splitImg(rotation)
            #     for img in split:
            #         hist = self.getHOG(img)
            #         hists.append(hist)
            # avgHist = self.getAvgHist(hists)
            roi = self.getROI(image.img)
            split = self.splitImg(roi)
            for im in split:
                hist = self.getHOG(im)
                # X.append(np.concatenate((hist, avgHist))) #AVG HIST 
                hists.append(hist)
            PCA_hists = self.pca.transform(hists)
            split_preds = self.clf.predict(PCA_hists)
            dist = self.clf.decision_function(PCA_hists)
            for i in range(0, len(split_preds)): sum += split_preds[i] 
            pred = sum/len(split)
            if(pred > 0.5): p = 1
            else: p = 0
            predictions.append([image.fileName, p, pred])
        return predictions #fileName, prediction, avg

    def makeCLFandPCA(self, present, notPresent):
        start = datetime.now()
        X = []
        y = []
        for image in present:
            rotations = self.getRotations(image.img)
            hists = []
            for rotation in rotations:
                split = self.splitImg(rotation)
                for img in split:
                    hist = self.getHOG(img)
                    hists.append(hist)
            # avgHist = self.getAvgHist(hists)
            for hist in hists:
                # X.append(np.concatenate((hist, avgHist)))
                X.append(hist)
                y.append(1)
        for image in notPresent:
            rotations = self.getRotations(image.img)
            hists = []
            for rotation in rotations:
                split = self.splitImg(rotation)
                for img in split:
                    hist = self.getHOG(img)
                    hists.append(hist)
            # avgHist = self.getAvgHist(hists)
            for hist in hists:
                # X.append(np.concatenate((hist, avgHist)))
                X.append(hist)
                y.append(0)
        self.clf = svm.SVC()
        if(X == [] or y == []): return
        X = np.array(X)
        print(X)
        self.pca = PCA(4)# adjust yourself
        self.pca.fit(X)
        X_train = self.pca.transform(X)
        print(X_train)
        self.clf.fit(X_train, y)
        getCLFTime = datetime.now()
        print('make CLF time: ' + str(getCLFTime - start))
        dump(self.clf, 'clf.pk1')
        dump(self.pca, 'pca.pk1')

    def readCLFandPCA(self):
        self.clf = load('clf.pk1')
        self.pca = load('pca.pk1')

    def getAcutance(self, image):
        if(type(image) == Image): img = self.getROI(image.img)
        else: img = image
        lbound = 25
        ubound = 150
        edges = cv2.Canny(img,lbound,ubound) 
        acutance = np.mean(edges)
        return acutance

    def focusImages(self, images):
        for image in images:
            acutance = self.getAcutance(image)
            if(acutance < self.acutance_thresh):
                image.img = self.wiener(image)

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
            predictions = self.predictPresence(testPres)
            print(predictions)
            for prediction in predictions:
                print(prediction)
                results_p.append(prediction)
            predictions = self.predictPresence(testNotPres)
            for prediction in predictions:
                print(prediction)
                results_np.append(prediction)
            i+=1
        path = os.path.join(self.package_dir, 'PCA4.txt')
        with open(path, 'w') as f:
            f.write('present\n')
            for prediction in results_p:
                f.write('%s %s %s\n' % (prediction[0], prediction[1], prediction[2]))
            f.write('not present\n')
            for prediction in results_np:
                f.write('%s %s %s\n' % (prediction[0], prediction[1], prediction[2]))        
        
    def getAvgHist(self, hists):
        sum = np.full(len(hists[0]), 0)
        for hist in hists:
            sum = np.add(sum, hist)
        divisor = np.full(len(hists[0]), len(hists))
        average = np.divide(sum, divisor)
        return average

    def writeHOGsToFile(self, images):
        averages = {}
        for image in images:
            hists = []
            rotations = self.getRotations(image.img)
            for rotation in rotations:
                split = self.splitImg(rotation)
                # split = self.splitImg(self.getROI(image.img))
                for im in split:
                    hist = self.getHOG(im)
                    hists.append(hist)
            average = self.getAvgHist(hists)
            averages[image.fileName] = average

        path = os.path.join(self.package_dir,'HOGs2/', 'avgHistsPR2.xlsx')
        wb = xlsxwriter.Workbook(path)
        sheet = wb.add_worksheet()
        sheet.write(0, 0, 'Bin')
        for k in range(1, 37):
            sheet.write(k, 0, k)

        col = 1
        for key in averages:
            hist = averages[key]
            sheet.write(0, col, key)
            for i in range(0, len(hist)):
                sheet.write(i+1, col, hist[i])
            col = col + 1
        wb.close()
        
    
    # Gaussian kernel generation function
    def creat_gauss_kernel(self, kernel_size=3, sigma=1, k=1):
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


    def wiener(self, image):
        img = image.img
        og = img
        before = self.getAcutance(img)  
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = color.rgb2gray(img)
        i = 5
        psf = self.creat_gauss_kernel(sigma=i)
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
        # np.set_printoptions(threshold=sys.maxsize)
        start = datetime.now()
        self.acutance_thresh = 5
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