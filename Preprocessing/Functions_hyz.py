from __future__ import print_function
import csv
from collections import defaultdict
import datetime
import sys
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import argmax


def column(matrix, i):
    col = [row[i] for row in matrix]
    count = 0
    for it in col:
        if it !='':
            count += 1
    return count

def findcol(matrix,i):
    return [row[i] for row in matrix]

def replacecol(col,matrix,i):
    for j in range(0,len(matrix)):
        matrix[j][i] = col[j]

def RepresentsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def date_to_num(year,reader,label):
    for k in year:
        datecol = findcol(reader,k)
        count = 0
        for i in range(0,len(datecol)):
            day = datecol[i]
            if count == 0:
                count = 1
                continue
            dayform = day.split('/')
            if len(dayform) != 3:
                reader[i][k] = label
            else:
                if int(dayform[0]) > 2000:
                    days = datetime.date(int(dayform[0]),int(dayform[1]),int(dayform[2]))
                else:
                    days = datetime.date(2000 + int(dayform[2]),int(dayform[0]),int(dayform[1]))
                iniday = datetime.date(2002,1,1)
                deltaday = days - iniday
                reader[i][k] = deltaday.days
    return reader

def ns_to_num(reader,label):
    nRows = len(reader)
    nColunms = len(reader[0])
    for j in range(nColunms):
        for i in range(nRows):
            if reader[i][j] == '':
                reader[i][j] = label
            if RepresentsFloat(reader[i][j]):
                reader[i][j] = float(reader[i][j])
    return reader

def label_to_num(reader,change):
    # change = []
    # nColunms = len(reader[0])
    # for i in range(nColunms):
    #     if not RepresentsFloat(reader[2][i]):
    #         change.append(i)
    for i in change:
        print("feature",i,"has been transformed")
        col = findcol(reader, i)
        count = 0
        num = 0
        dic = {}
        for j in range(0, len(col)):
            if num == 0:
                num = 1
                continue
            if col[j] == '':
                col[j] = 9999
                continue
            if col[j] in dic:
                col[j] = dic[col[j]]
            else:
                dic[col[j]] = count
                col[j] = count
                count += 1
            replacecol(col, reader, i)
    return reader

def find_label_index(reader):
    change = []
    nColunms = len(reader[0])
    for i in range(nColunms):
        if not RepresentsFloat(reader[2][i]):
            change.append(i)
    print("label features are:", change)
    return change

def hot_encoding(reader, change):
    for i in change:
        data=[]
        for j in range(len(reader)):
            data.append(reader[j][i])
        data = array(data)
        onehot_encoder = OneHotEncoder(sparse=False)
        data = data.reshape(len(data), 1)
        onehot_encoded = onehot_encoder.fit_transform(data)
        if (len(onehot_encoded[0])) > 10:
            print(i,"column has", len(onehot_encoded[0]),"features")
        for j in range(len(reader)):
            for k in range(len(onehot_encoded[0])):
                reader[j].append(onehot_encoded[j][k])
    return reader