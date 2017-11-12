import csv
import datetime
import sys
from parsing2 import parsing

def run():
	#firstly read the target file
	# The reqiurement of the input file
	# inputdata must firstly be sorted by date incremently,then by id incremently
	# target data must be sorted by id incremently
	csvfile1 = open('LOCF_data.csv', 'rb');
	inputfeature = list(csv.reader(csvfile1));
	csvfile2 = open('LOCF_train.csv','rb');
	traindata = list(csv.reader(csvfile2));
	csvfile3 = open('LOCF_test.csv','rb');
	testdata = list(csv.reader(csvfile3));
	parsing(inputfeature,traindata,testdata);











if __name__ == '__main__':
	run();