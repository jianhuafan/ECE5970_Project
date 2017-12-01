import numpy as np
import csv

def max_length(input_path):
	i = max_length = 0
	input_data = input_parse(input_path)
	while i < input_data.shape[0]:
		first = input_data[i, 0]
		start = i
		while i < input_data.shape[0] and input_data[i, 0] == first:
			i += 1
		end = i
		max_length = max(max_length, end - start)
	return max_length



def input_parse(fpath):
	with open(fpath, 'r') as f:
		data = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
	data = np.asarray(data, dtype=np.float32)
	return data

def output_parse(fpath):
	with open(fpath, 'r') as f:
		data = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
	data = np.asarray(data, dtype=np.float32)
	return data

def reg_sequences(index, batch_size, sequence_length, input_path, output_path): #index = 3, 4, 5
	input_data = input_parse(input_path)
	output_data = output_parse(output_path)
	real_input_data = np.delete(input_data, 0, axis=1)
	real_output_data = np.delete(output_data, 0, axis=1)
	feature_size = real_input_data.shape[1]
	num_target = 1
	input_index = 0
	try:
		for i in range(0, output_data.shape[0], batch_size):
			if i + sequence_length + batch_size >= output_data.shape[0]:
				# effective_batch_size = output_data.shape[0] - sequence_length - i
				effective_batch_size = -1
			else:
				effective_batch_size = batch_size
			if effective_batch_size <= 0:
				X, y = None, None
			else:
				X = np.zeros(
					(effective_batch_size, sequence_length, feature_size),
					dtype=np.float32
				)
				y = np.zeros((effective_batch_size, num_target), dtype=np.float32) ####??????
			if X is not None and y is not None:
				for j in range(effective_batch_size):
					pid = output_data[i + j][0]
					start_index = input_index
					if input_data[input_index][0] == pid:
						while input_index < input_data.shape[0] and input_data[input_index][0] == pid:
							input_index += 1
						end_index = input_index
						X[j, np.arange(end_index-start_index)] = real_input_data[start_index:end_index]
						y[j] = real_output_data[i + j, index]
					else:
						input_index += 1
			yield X, y
	except:
		print('sequences generation is stopped!')



def sequences(batch_size, sequence_length, input_path, output_path):
	input_data = input_parse(input_path)
	output_data = output_parse(output_path)
	real_input_data = np.delete(input_data, 0, axis=1)
	real_output_data = np.delete(output_data, 0, axis=1)
	feature_size = real_input_data.shape[1]
	num_target = 3
	input_index = 0
	try:
		for i in range(0, output_data.shape[0], batch_size):
			if i + sequence_length + batch_size >= output_data.shape[0]:
				# effective_batch_size = output_data.shape[0] - sequence_length - i
				effective_batch_size = -1
			else:
				effective_batch_size = batch_size
			if effective_batch_size <= 0:
				X, y = None, None
			else:
				X = np.zeros(
					(effective_batch_size, sequence_length, feature_size),
					dtype=np.float32
				)
				y = np.zeros((effective_batch_size, num_target), dtype=np.float32) ####??????
			if X is not None and y is not None:
				for j in range(effective_batch_size):
					pid = output_data[i + j][0]
					start_index = input_index
					if input_data[input_index][0] == pid:
						while input_index < input_data.shape[0] and input_data[input_index][0] == pid:
							input_index += 1
						end_index = input_index
						X[j, np.arange(end_index-start_index)] = real_input_data[start_index:end_index]
						y[j] = real_output_data[i + j, :3]
					else:
						input_index += 1
			yield X, y
	except:
		print('sequences generation is stopped!')
	    
	    
	# finally:
		# print('sequences generated successfully!')



def test_sequences():
	input_path = 'data/input_train_other.csv'
	output_path = 'data/target_train.csv'
	infor_path = 'data/infor_train.csv'
	max_sequence_length = max_length(input_path)
	max_sequence_length = 14
	batch_size = 1647
	seq_iter = other_sequences(batch_size, max_sequence_length, input_path, output_path, infor_path)
	count = 0
	for (X, y) in seq_iter:
		if X is not None and y is not None:
			count += 1
			# for i in range(X.shape[0]):
			# 	if i == 0:
					# print('X = %s, y = %s' % (
					# 	X[i], y[i]
					# ))
	print(count)


if __name__ == '__main__':
	test_sequences()
	

