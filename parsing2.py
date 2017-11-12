import csv
import datetime
import sys



def parsing(inputfeature,traindata,testdata):
	inputfeaturetrain = []
	inputfeaturetest = []
	trainkey = []
	testkey = []
	for row in traindata:
		trainkey.append(row[1]);
	for row in testdata:
		testkey.append(row[1]);

	for i in range(0,len(inputfeature)):
		inputfeature[i][0] = str(int(float(inputfeature[i][0])));

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
	count = 0;
	for j in range(0,len(traindata)):
		if j >= 1 and traindata[j][1] == traindata[j - 1][1]:
			count += 1;
		else:
			count = 0;
		for i in range(0,len(inputfeaturetrain)):
			if inputfeaturetrain[i][0] == traindata[j][1]:
				dt = float(traindata[j][0]) - float(inputfeaturetrain[i][1]);
				new = []
				#new2 = []
				new.append(int(float(inputfeaturetrain[i][0]))*10 + count);
				new.append(dt);#replace dt with t
				#new2.append(traindata[j][1]);
				for k in range(2,len(inputfeaturetrain[0])):
					new.append(inputfeaturetrain[i][k]);
				# for m in range(2,len(traindata[0])):
				# 	new2.append(traindata[j][m]);
				finalinputtrain.append(new);
				# finaloutputtrain.append(new2);
		new2  = []
		new2.append(int(float(traindata[j][1]))*10 + count);
		for m in range(2,len(traindata[0])):
			new2.append(traindata[j][m]);
		finaloutputtrain.append(new2);

	count = 0;
	for j in range(0,len(testdata)):
		if j >= 1 and testdata[j][1] == testdata[j - 1][1]:
			count += 1;
		else:
			count = 0;
		for i in range(0,len(inputfeaturetest)):
			if inputfeaturetest[i][0] == testdata[j][1]:
				dt = float(testdata[j][0]) - float(inputfeaturetest[i][1]);
				new = []
				#new2 = []
				new.append(int(float(inputfeaturetrain[i][0]))*10 + count);
				new.append(dt);#replace dt with t
				#new2.append(traindata[j][1]);
				for k in range(2,len(inputfeaturetest[0])):
					new.append(inputfeaturetest[i][k]);
				# for m in range(2,len(traindata[0])):
				# 	new2.append(traindata[j][m]);
				finalinputtest.append(new);
				# finaloutputtrain.append(new2);
		new2  = []
		new2.append(int(float(testdata[j][1]))*10 + count);
		for m in range(2,len(testdata[0])):
			new2.append(testdata[j][m]);
		finaloutputtest.append(new2);



	file = open('finaloutputtrain.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(finaloutputtrain);
	finally:
		file.close();

	file = open('finalinputtrain.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(finalinputtrain);
	finally:
		file.close();

	file = open('finaloutputtest.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(finaloutputtest);
	finally:
		file.close();

	file = open('finalinputtest.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(finalinputtest);
	finally:
		file.close();

