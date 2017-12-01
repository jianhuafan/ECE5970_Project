import csv
import datetime
import sys
from parsing2 import parsing

from Functions_hyz import date_to_num, ns_to_num, label_to_num, find_label_index, hot_encoding
from Functions_yfx import delete_sparse, select_sparse, delete_colunms, locf, pca, locf_pre, sum_col, sca


def run():
    csvfile1 = open('332222.csv', 'r')
    inputfeature = list(csv.reader(csvfile1))
    csvfile2 = open('newtrain.csv', 'r')
    traindata = list(csv.reader(csvfile2))
    csvfile3 = open('newtest.csv', 'r')
    testdata = list(csv.reader(csvfile3))
    # testdata = to_float(testdata)
    parsing(inputfeature, traindata, testdata)


if __name__ == '__main__':
    run()
