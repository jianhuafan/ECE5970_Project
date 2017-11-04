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
    col = [row[i] for row in matrix];
    count = 0;
    for it in col:
    	if it !='':
    		count += 1;
    return count;

def findcol(matrix,i):
	return [row[i] for row in matrix];

def replacecol(col,matrix,i):
	for j in range(0,len(matrix)):
		matrix[j][i] = col[j];



def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

with open('TADPOLE_InputData.csv', 'rb') as csvfile:
	reader = list(csv.reader(csvfile));
	#count the number of in null value;
	# for i in range(0,len(reader[1])):
	# 	#print len(column(reader,i));
	# 	print column(reader,i);

	#delete 456
	# for i in range(len(reader)):
	# 	del reader[i][456];

	# change the date of column year into number of day with date distance to 
	# 2006/1/1. Whether this method is acceptable is to be more considered.

	
	year = [1,46,85,86,91,456,457,462,820,823,1881,1886]
	for k in year:
		datecol = findcol(reader,k);
		count = 0;
		for i in range(0,len(datecol)):
			day = datecol[i];
			if count == 0:
				count = 1;
				continue;
		 	dayform = day.split('/');
		 	if len(dayform) != 3:
		 		reader[i][k] = 9999;
		 	else:
		 		if int(dayform[0]) > 2000:
					days = datetime.date(int(dayform[0]),int(dayform[1]),int(dayform[2]));
				else:
					days = datetime.date(2000 + int(dayform[2]),int(dayform[0]),int(dayform[1]));
				iniday = datetime.date(2002,1,1);
				deltaday = days - iniday;
				
				reader[i][k] = deltaday.days;
	for row  in reader:
		for i in range(0,len(row)):
			if row[i] == '':
				row[i] = 9999;
			if RepresentsFloat(row[i]):
				row[i] = float(row[i]);

	change = []


	for i in range(0,len(reader[0])):
		if not RepresentsFloat(reader[2][i]):
			change.append(i);


	#for some string value, we automatically give 
	#some increment int value for them
	for i in change:
		col = findcol(reader,i);
		count = 0;
		num = 0;
		dic = {};
		for j in range(0,len(col)):
			if num == 0:
				num = 1;
				continue;
			if col[j] == '':
				col[j] = 9999;
				continue;
			if col[j] in dic:
				col[j] = dic[col[j]];
			else:
				dic[col[j]] = count;
				col[j] = count;
				count += 1;
		replacecol(col,reader,i);




	
	#change all value which value is not time from string to value;


	# delete 0 row
	del reader[0];
	

	#firstly doing the data imputation
	# imp = Imputer(missing_values = -1000,strategy ='median',axis = 0)
	# imp.fit([reader[0],reader[1],reader[3]]);
	# imp.transform(reader);

	#do hotencoding for the data, since we haven't have imputation here, so basically missing data are taken as one data
	#and we will appnd data to the end of initial data
	for i in change:
		data=[]
		for j in range(len(reader)):
			data.append(reader[j][i])
		data = array(data)
		onehot_encoder = OneHotEncoder(sparse=False)
		data = data.reshape(len(data), 1)
		onehot_encoded = onehot_encoder.fit_transform(data)
		for j in range(len(reader)):
			for k in range(len(onehot_encoded[0])):
				reader[j].append(onehot_encoded[j][k]);

	#for those hot encoding, delete the initial data
	newdata = []
	for i in range(len(reader)):
		new = []
		for j in range(len(reader[0])):
			if j not in change:
				new.append(reader[i][j]);
		newdata.append(new);

	
	file = open('inputdata1.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(newdata);
	finally:
		file.close();











	#print (columns['update_stamp_DTIROI_04_30_14']);


