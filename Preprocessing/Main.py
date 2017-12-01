import csv
from Functions_hyz import date_to_num,ns_to_num,label_to_num,find_label_index,hot_encoding
from Functions_yfx import delete_sparse, select_sparse,delete_colunms,locf,pca, locf_pre,sum_col, sca, to_float
import numpy as np


with open('TADPOLE_TargetData_valid.csv', 'r') as csvfile:
# with open('input.csv', 'r') as csvfile:
    label = 9999999  ## a label for null in input file
    data = list(csv.reader(csvfile))
    
    # data = date_to_num(data)
    # print("date_to_num set")
    # # delete all other year date
    # delete = [46, 85, 86, 91, 456, 457, 462, 820, 823, 1881, 1886]
    # data = delete_colunms(data,delete)
    # # delete sparse features
    # data = delete_sparse(data,70) ## 375 features
    # # data = delete_sparse(data,72) ## supposed to be 715 features
    # print("delete_data set")
    # # transfrom null to label & transform str to float
    # data = ns_to_num(data,label)
    # print("ns_to_num set")
    # # transform labels to number
    # print("length of data is:",len(data[0]))
    # change = find_label_index(data)
    # data = label_to_num(data,change)
    #
    # del data[0]
    # # # change = [2, 3, 5, 7, 8, 9, 27, 52, 369, 370, 371, 372, 373, 374, 375, 376, 377]
    # # # # changr = [2, 5, 7, 8, 9, 27, 52, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 707]
    # data = to_float(data)
    # # # locf
    # data = locf(data,label)
    # data = sca(data)
    # # # # hot-encoding
    # # # data = hot_encoding(data,change)

    # del data[0]
    # data = to_float(data)
    # data = pca(data, 0.8)

    # print("length of data", len(data[0]))
    file = open('target_valid.csv', 'w')
    try:
        writer = csv.writer(file)
        writer.writerows(data)
    finally:
        file.close()
