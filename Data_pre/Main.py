import csv
from Functions_hyz import date_to_num,ns_to_num,label_to_num,find_label_index,hot_encoding
from Functions_yfx import delete_sparse, select_sparse,delete_colunms,locf,pca, locf_pre,sum_col, sca

with open('TADPOLE_InputData.csv', 'r') as csvfile:
# with open('input.csv', 'r') as csvfile:
    label = 9999  ## a label for null in input file
    data = list(csv.reader(csvfile))

    # date_to_num
    year = [1,46, 85, 86, 91, 456, 457, 462, 820, 823, 1881, 1886]
    data = date_to_num(year, data)
    print("date_to_num set")

    # delete all other year date
    delete = [46, 85, 86, 91, 456, 457, 462, 820, 823, 1881, 1886]
    data = delete_colunms(data,delete)

    # delete sparse features
    data = delete_sparse(data,41) ## supposed to be 715 features
    print("delete_data set")

    # transfrom null to label & transform str to float
    data = ns_to_num(data,label)
    print("ns_to_num set")

    # transform labels to number
    # print(len(data[0]))
    change = find_label_index(data)
    data = label_to_num(data,change)
    D = [707]
    data = delete_colunms(data,D)
    # change = [2, 5, 7, 8, 9, 27, 52, 369, 370, 371, 372, 373, 374, 375, 376, 377, 707]
    # data = locf_pre(data)

    # data = locf(data,label)
    # data = hot_encoding(data,change)

    # delete = []
    # for j in range(700,len(data[0])):
    #     if sum_col(data,j) == 0:
    #         delete.append(j)
    # print(j)
    # data = delete_colunms(data, delete)

    # data = sca(data)

    # print("length of data", len(data[0]))
    file = open('input_temp.csv', 'w')
    try:
        writer = csv.writer(file)
        writer.writerows(data)
    finally:
        file.close()