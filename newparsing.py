import csv
import datetime
import sys


# This py file is responsible for spiltting the input file into
# the file train and test one. also I do the crossproduct in this file

with open('LOCF_data.csv', 'rb') as csvfile1:
	inputfeature = list(csv.reader(csvfile1));
	csvfile2 = open('newtrain1.csv','rb');
	traindata = list(csv.reader(csvfile2));
	csvfile3 = open('newtest1.csv','rb');
	testdata = list(csv.reader(csvfile3));
	inputfeaturetrain = []
	inputfeaturetest = []
	trainkey = []
	testkey = []
	for row in traindata:
		trainkey.append(row[1]);
	for row in testdata:
		testkey.append(row[1]);
	print testkey;

	for i in range(0,len(inputfeature)):
		#if key in train data, write to inputfeaturetrain
		if inputfeature[i][0] in trainkey:
			new = []
			for j in range(len(inputfeature[0])):
				new.append(inputfeature[i][j]);
			inputfeaturetrain.append(new);
		if inputfeature[i][0] in testkey:
			new = []
			for j in range(len(inputfeature[0])):
				new.append(inputfeature[i][j]);
			inputfeaturetest.append(new);

	finalinputtrain = []
	finaloutputtrain = []
	finalinputtest = []
	finaloutputtest = []

	for i in range(0,len(inputfeaturetrain)):
		for j in range(0,len(traindata)):
			if inputfeaturetrain[i][0] == traindata[j][1]:
				dt = float(traindata[j][0]) - float(inputfeaturetrain[i][1]);
				new = []
				new2 = []
				new.append(inputfeaturetrain[i][0]);
				new.append(dt);#replace dt with t
				new2.append(traindata[j][1]);
				for k in range(2,len(inputfeaturetrain[0])):
					new.append(inputfeaturetrain[i][k]);
				for m in range(2,len(traindata[0])):
					new2.append(traindata[j][m]);
				finalinputtrain.append(new);
				finaloutputtrain.append(new2);

	for i in range(0,len(inputfeaturetest)):
		for j in range(0,len(testdata)):
			if inputfeaturetest[i][0] == testdata[j][1]:
				dt = float(testdata[j][0]) - float(inputfeaturetest[i][1]);
				new = []
				new2 = []
				new.append(inputfeaturetest[i][0]);
				new.append(dt);#replace dt with t
				new2.append(testdata[j][1]);
				for k in range(2,len(inputfeaturetest[0])):
					new.append(inputfeaturetest[i][k]);
				for m in range(2,len(testdata[0])):
					new2.append(testdata[j][m]);
				finalinputtest.append(new);
				finaloutputtest.append(new2);

	for i in range(len(finaloutputtrain)):
		del finaloutputtrain[0];


	file = open('finaloutputtrain1.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(finaloutputtrain);
	finally:
		file.close();

	file = open('finalinputtrain1.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(finalinputtrain);
	finally:
		file.close();

	file = open('finaloutputtest1.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(finaloutputtest);
	finally:
		file.close();

	file = open('finalinputtest1.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(finalinputtest);
	finally:
		file.close();