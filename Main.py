import csv
from Functions_Hyz import date_to_num,ns_to_num,label_to_num,find_label_index,hot_encoding
from Functions import delete_sparse, select_sparse,delete_colunms,locf,pca

# with open('TADPOLE_InputData.csv', 'r') as csvfile:
with open('test.csv', 'r') as csvfile:
    label = 9999  ## a label for null in input file
    data = list(csv.reader(csvfile))

#     # data = locf(data,label)
#     # date_to_num
#     year = [1,46, 85, 86, 91, 456, 457, 462, 820, 823, 1881, 1886]
#     data = date_to_num(year, data)
#     print("date_to_num set")
#     print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#
#
#     # delete all other year date
#     delete = [46, 85, 86, 91, 456, 457, 462, 820, 823, 1881, 1886]
#     data = delete_colunms(data,delete)
#
#     # delete sparse features
#     data = delete_sparse(data,41) ## supposed to be 715 features
#     print("delete_data set",strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#
#     # transfrom null to label & str to float
#     data = ns_to_num(data,label)
#     print("ns_to_num set")
#     print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#
#     # transform labels to number
#
#     change = find_label_index(data)
#     data = label_to_num(data,change)

    # change = [2, 5, 7, 8, 9, 27, 52, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 707]
    #
    # data = locf(data,label)
    #
    # data = hot_encoding(data,change)

    data = pca(data,0.9)






    file = open('PCA.csv', 'w')
    try:
        writer = csv.writer(file)
        writer.writerows(data)
    finally:
        file.close()