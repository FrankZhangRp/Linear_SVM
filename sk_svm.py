import numpy as np
import time
from sklearn import svm
from mnist import load_test, load_train, load_binary_2
# 数据预处理 将数据变成矩阵
train_dic = load_train()
test_dic = load_test()
train_data = train_dic['images'].reshape(60000,-1)
train_label = train_dic['labels']

test_data = test_dic['images'].reshape(10000, -1)
test_label = test_dic['labels']

def label_tune(labels, choose_num=0):
    labels_size = np.shape(labels)
    tune_labels = np.empty(labels_size)
    for i in range(len(labels)):
        if labels[i] == choose_num:
            tune_labels[i] = 1
        else:
            tune_labels[i] = -1
    return tune_labels

def ovo():
    for i in range(10):
        for j in range(10):
            if j<=i:
                continue
            print(i,j, end='\t')
            train_features, train_labels, test_features, test_labels = binary_data(i,j)
            lsvm = svm.SVC(C=100.0, kernel='linear')
            lsvm.fit(train_features, train_labels)
            predictions = [int(a) for a in lsvm.predict(test_features)]
            num_correct = sum(int(a==y) for a,y in zip(predictions, test_labels))
            print(num_correct/float(len(test_features)))

def binary_data(num1, num2):
    data_dic = load_binary_2(num1, num2)
    train_features = data_dic['train_data']
    train_labels = data_dic['train_label']
    test_features = data_dic['test_data']
    test_labels = data_dic['test_label']
    train_features = train_features.reshape(len(train_features), -1)
    test_features = test_features.reshape(len(test_features), -1)
    return train_features, train_labels, test_features, test_labels

if __name__ == '__main__':
    # ovo()
    start_time = time.time()
    
    data_dic = load_binary_2(0, 6)
    train_features = data_dic['train_data']
    train_labels = data_dic['train_label']
    test_features = data_dic['test_data']
    test_labels = data_dic['test_label']
    train_features = train_features.reshape(len(train_features), -1)
    test_features = test_features.reshape(len(test_features), -1)
    
    # train_dic = load_train()
    # test_dic = load_test()
    # train_features = train_dic['images'].reshape(60000, -1)
    # train_labels = train_dic['labels']
    # test_features = test_dic['images'].reshape(10000, -1)
    # test_labels = test_dic['labels']
    # train_labels = label_tune(labels=train_labels, choose_num=0)
    # test_labels = label_tune(labels=test_labels, choose_num=0)


    lsvm = svm.SVC(C=10000.0, kernel='linear', decision_function_shape='ovr')
    lsvm.fit(train_features, train_labels)
    predictions = [int(a) for a in lsvm.predict(test_features)]
    num_correct = sum(int(a==y) for a,y in zip(predictions, test_labels))
    print(num_correct/float(len(test_features)))
    end_time = time.time()
    print(end_time-start_time)