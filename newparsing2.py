import csv
import datetime
import sys


# this file takes charge of spiliting the output file from 6 column to 4 files each contains seperate
#variable that I want to predict. The newtest1 contains 3 predicted columns in [1,0,0] form. Others contains
# one column
with open('newtest.csv', 'rb') as csvfile1:
	reader = list(csv.reader(csvfile1));

	newone = []
	for i in range(len(reader)):
		if reader[i][2] == '-1000' or reader[i][3] == '-1000' or reader[i][4] == '-1000':
			continue;
		else:
			new = []
			for j in range(0,5):
				new.append(reader[i][j]);
			newone.append(new);


	newtwo = []
	for i in range(len(reader)):
		if reader[i][5] == '-1000':
			continue;
		else:
			new = []
			new.append(reader[i][0])
			new.append(reader[i][1])
			new.append(reader[i][5])
			newtwo.append(new);

	newthree = []
	for i in range(len(reader)):
		if reader[i][6] == '-1000':
			print("find!");
			continue;
		else:
			new = []
			new.append(reader[i][0])
			new.append(reader[i][1])
			new.append(reader[i][6])
			newthree.append(new);


	newfour = []
	for i in range(len(reader)):
		if reader[i][7] == '-1000':
			continue;
		else:
			new = []
			new.append(reader[i][0])
			new.append(reader[i][1])
			new.append(reader[i][7])
			newfour.append(new);

	file = open('newtest1.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(newone);
	finally:
		file.close();

	file = open('newtest2.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(newtwo);
	finally:
		file.close();

	file = open('newtest3.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(newthree);
	finally:
		file.close();

	file = open('newtest4.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(newfour);
	finally:
		file.close();




	