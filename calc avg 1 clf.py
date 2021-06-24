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
    path = os.path.join(package_dir, 'PCA4.txt')
    print(path)
    with open(path) as f:
        flines = f.readlines()
        lines = [line.strip() for line in flines]
        sumTotal = 0
        iTotal = 0
        i=0
        sum = 0
        truePres = 0 #accurate pres predictions
        falsePres = 0
        trueNotPres = 0
        falseNotPres = 0
        j = 0
        while(j < len(lines)):
            line = lines[j]
            if(line ==  'present'):
                j+=1
                while(j < len(lines) and lines[j] != 'not present'):
                    line = lines[j]
                    j += 1
                    arr = line.split()
                    avg = float(arr[len(arr)-1])
                    pred = int(arr[len(arr)-2])
                    sum += avg
                    sumTotal += avg
                    iTotal += 1
                    i+=1
                    if(pred == 0): falseNotPres+=1
                    elif(pred == 1): truePres+=1
            if(lines[j] == 'not present'):
                j+=1
                while(j < len(lines) and 'not_present' in lines[j]):
                    line = lines[j]
                    j += 1
                    arr = line.split()
                    avg = float(arr[len(arr)-1])
                    pred = int(arr[len(arr)-2])
                    sum += 1 - avg
                    sumTotal += 1 - avg
                    iTotal += 1
                    i+=1
                    if(pred == 0): trueNotPres+=1
                    else: falsePres+=1
                avg = sum/i
                print('truePres = %s\nfalsePres = %s\ntrueNotPres = %s\nfalseNotPres = %s\navg = %s\n' 
                % (truePres, falsePres, trueNotPres, falseNotPres, avg))
        avg = sumTotal/iTotal
if __name__ == '__main__':
    main()