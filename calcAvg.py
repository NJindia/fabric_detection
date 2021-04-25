import cv2
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from sklearn import svm
from joblib import load, dump
from datetime import datetime

def main():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(package_dir, 'k-fold purple.txt')
    with open(path) as f:
        sum = 0
        i=0
        sumBinary = 0
        flines = f.readlines()
        lines = [line.strip() for line in flines]
        sumPPV = 0
        sumNPV = 0
        negI = 0
        posI = 0

        for j in range(0, len(lines)):
            line = lines[j]
            j += 1
            if(line ==  'present'):
                while(line != 'not present' and j < len(lines)):
                    line = lines[j]
                    j += 1
                    if 'avg = ' in line:
                        i+=1
                        posI+=1
                        sum += float(line[6:])
                        sumPPV+= float(line[6:])
                    elif 'present prediction:' in line:
                        ind = line.index('present prediction: ')
                        index = ind+len('present prediction: ')
                        sumBinary+=int(line[index:index+1])
            elif(line == 'not present'):
                while(line != 'present' and j < len(lines)):
                    line = lines[j]
                    j += 1
                    if 'avg = ' in line:
                        i+=1
                        negI+=1
                        sum += 1 - float(line[6:])
                        sumNPV+= 1 - float(line[6:])
                    elif 'not present prediction: ' in line:
                        ind = line.index('not present prediction: ')
                        index = ind+len('not present prediction: ')
                        sumBinary+=1 - int(line[index:index+1])
        avg = sum/i
        binAvg = sumBinary/i
        PPV = sumPPV/posI
        NPV = sumNPV/negI
        print(sum, sumBinary, i)
        print('PPV = ' + str(PPV))
        print('NPV = ' + str(NPV))
        print('avg = ' + str(avg))
        print('binary avg = ' + str(binAvg))
if __name__ == '__main__':
    main()