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
    with open('k-fold present.txt') as f:
        sum = 0
        i=0
        sumBinary = 0
        for line in f:
            if 'avg = ' in line:
                i+=1
                sum += float(line[6:])
            elif 'present prediction:' in line:
                ind = line.index('present prediction: ')
                index = ind+len('present prediction: ')
                sumBinary+=int(line[index:index+1])
        avg = sum/i
        binAvg = sumBinary/i
        print('avg = ' + str(avg))
        print('binary avg = ' + str(binAvg))
    
if __name__ == '__main__':
    main()