import csv
import numpy as np
import datetime
from sklearn.preprocessing import Imputer

# this function takes charge of do some preprocessing for the train data
# like date change and imputation
def findcol(matrix,i):
	return [row[i] for row in matrix];

def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

def isNaN(num):
    return num != num


with open('TADPOLE_TargetData_train.csv', 'rb') as csvfile:
	reader = list(csv.reader(csvfile));
	del reader[0];
	
	
	datecol = findcol(reader,0);
	# change date form
	for i in range(0,len(datecol)):
		day = datecol[i];
	 	dayform = day.split('/');
	 	if len(dayform) != 3:
	 		datecol[i] = -1;
	 	else:
	 		days = datetime.date(2000 + int(dayform[2]),int(dayform[0]),int(dayform[1]));
	 		iniday = datetime.date(2002,1,1);
	 		deltaday = days - iniday;
			
			reader[i][0] = deltaday.days;
	# #change all string to int value
	for row  in reader:
		for i in range(0,len(row)):
			if row[i] == 'NAN':
				row[i] = -1000;
			if RepresentsFloat(row[i]):
				row[i] = float(row[i]);
	# there are some value with 5 or 6 nan, we just delete this row because
	# it is impossible to do the imputation in this situation.
	newdata = [];
	for i in range(0,len(reader)):
		count = 0;
		for j in range(0,8):
			if isNaN(reader[i][j]):
				count += 1;
		if count < 5:
			new = [];
			for j in range(0,8):
				if isNaN(reader[i][j]):
					new.append(-1000);
				else:
					new.append(reader[i][j]);
			newdata.append(new);

	#doing data imputation based on the column value

	file = open('newtrain.csv','wt');
	try:
		writer = csv.writer(file);
		writer.writerows(newdata);
	finally:
		file.close();

	
			





