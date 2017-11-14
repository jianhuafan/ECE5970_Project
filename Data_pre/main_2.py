import csv
import datetime
import sys
from parsing2 import parsing

def run():
    csvfile1 = open('input.csv', 'r')
    inputfeature = list(csv.reader(csvfile1))
    csvfile2 = open('target_train.csv','r')
    traindata = list(csv.reader(csvfile2))
    csvfile3 = open('target_test.csv','r')
    testdata = list(csv.reader(csvfile3))
    parsing(inputfeature,traindata,testdata)

if __name__ == '__main__':
	run()