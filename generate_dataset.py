# encoding=utf8
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
N = 10 #生成训练数据的个数

# AX=0 相当于matlab中 null(a','r')
def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()

# 符号函数，之后要进行向量化
def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    elif x < 0:
        return -1
#noisy=False，那么就会生成N的dim维的线性可分数据X，标签为y
#noisy=True, 那么生成的数据是线性不可分的,标签为y
def mk_data(N, noisy=False):
    rang = [-10,10]
    dim = 2

    X=np.random.rand(dim,N)*(rang[1]-rang[0])+rang[0]

    while True:
        Xsample = np.concatenate((np.ones((1,dim)), np.random.rand(dim,dim)*(rang[1]-rang[0])+rang[0]))
        k,w=null(Xsample.T)
        y = sign(np.dot(w.T,np.concatenate((np.ones((1,N)), X))))
        if np.all(y):
            break

    if noisy == True:
        idx = random.sample(range(1,N), N/10)

        for id in idx:
            y[0][id] = -y[0][id]

    return (X,y,w)

def data_visualization(X,y,title):
    class_1 = [[],[]]
    class_2 = [[],[]]

    size = len(y)

    for i in range(size):
        X_1 = X[0][i]
        X_2 = X[1][i]

        if y[i] == 1:
            class_1[0].append(X_1)
            class_1[1].append(X_2)
        else:
            class_2[0].append(X_1)
            class_2[1].append(X_2)


    plt.figure(figsize=(8, 6), dpi=80)
    plt.title(title)

    axes = plt.subplot(111)

    type1 = axes.scatter(class_1[0], class_1[1], s=20, c='red')
    type2 = axes.scatter(class_2[0], class_2[1], s=20, c='green')


    plt.show()

def rebuild_features(features):
    size = len(features[0])

    new_features = []
    for i in range(size):
        new_features.append([features[0][i],features[1][i]])

    return new_features

def generate_dataset(size, noisy = False, visualization = True):
    global sign
    sign = np.vectorize(sign)
    X,y,w = mk_data(size,False)
    y = list(y[0])

    if visualization:
        data_visualization(X,y,'all data')         #数据可视化

    testset_size = int(len(y)*0.1)

    indexes = [i for i in range(len(y))]
    test_indexes = random.sample(indexes,testset_size)
    train_indexes = list(set(indexes)-set(test_indexes))

    trainset_features = [[],[]]
    trainset_labels = []

    testset_features = [[],[]]
    testset_labels = []

    for i in test_indexes:
        testset_features[0].append(X[0][i])
        testset_features[1].append(X[1][i])
        testset_labels.append(y[i])


    if visualization:
        data_visualization(testset_features,testset_labels,'test set')

    for i in train_indexes:
        trainset_features[0].append(X[0][i])
        trainset_features[1].append(X[1][i])
        trainset_labels.append(y[i])

    if visualization:
        data_visualization(trainset_features,trainset_labels,'train set')

    return rebuild_features(trainset_features),trainset_labels,rebuild_features(testset_features),testset_labels

def plot_decision_regions(X, y, classifier,test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'cyan','gray')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    X = np.array(X)
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    #X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], c='',
                alpha=1.0, linewidth=1, marker='o',
                s=55, label='test set')


if __name__ == '__main__':

    size = 1000
    train_features, train_labels, test_features, \
    test_labels=generate_dataset(size)
    from sklearn import svm
    lsvm =  svm.SVC(C=10000.0, kernel='linear')
    lsvm.fit(train_features, train_labels)
    predictions = [int(a) for a in lsvm.predict(test_features)]
    num_correct = sum(int(a==y) for a,y in zip(predictions, test_labels))
    print(num_correct/float(len(test_features)))
    plot_decision_regions(train_features, train_labels, classifier=lsvm)
    plt.show()
    # generate_dataset
    # print sign
    # sign = np.vectorize(sign)
    # X,y,w = mk_data(size,False)
    #
    # data_visualization(X,y)
# encoding=utf8