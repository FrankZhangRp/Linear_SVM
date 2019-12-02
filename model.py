# linear svm模型
import numpy as np
import logging

class LSVM(object):
    def __init__(self, epsilon=0.001):
        # 精度
        self.epsilon = epsilon

    def _init_parameters(self, input_data, input_label):    
        '''X和Y都必须是numpy的数组，其中input_data是二维向量[N,n]，input_label是二值label{-1,+1}'''
        self.X = input_data
        self.Y = input_label
        # 初始化SVM参数
        self.b = 0.0
        self.n = len(input_data[0])
        self.N = len(input_data)
        
        # 这里使用alpha代替文章中的lambda
        self.alpha = [0.0] * self.N #初始化a(0)全为0
        self.E = [self._E_(i) for i in range(self.N)] # 得到E值的表
        
        # 模型超参数
        self.C = 100
        self.Max_Interation = 5000


    def _dot_(self,x1,x2):
        # 计算两个向量的内积 
        return sum([x1[k] * x2[k] for k in range(self.n)])
        # 使用numpy实现内积
        # return np.dot(x1.reshape(-1), x2.reshape(-1))

    def _g_(self, i):
        # 计算公式24的停机条件
        result = self.b
        for j in range(self.N):
            result += self.alpha[j] * self.Y[j]*(self._dot_(self.X[j], self.X[i]))
        return result

    def _E_(self, i):
        # 计算E值 是求解子问题的一个步骤
        return self._g_(i) - self.Y[i]

    def try_E(self,i):
        result = self.b - self.Y[i]
        for j in range(self.N):
            if self.alpha[j]<0 or self.alpha[j]>self.C:
                continue
            result += self.Y[j]*self.alpha[j]*self._dot_(self.X[i], self.X[j])
        return result

    def _satisft_KKT(self, i):
        # 用来计算第i个alpha是否满足KKT条件 在精度条件下
        ygx = self.Y[i] * self._g_(i)
        if abs(self.alpha[i])<self.epsilon:
            return ygx > 1 or ygx == 1
        elif abs(self.alpha[i]-self.C)<self.epsilon:
            return ygx < 1 or ygx == 1
        else:
            return abs(ygx-1) < self.epsilon
    
    def is_stop(self):
        # 用来判断是否满足停机条件
        for i in range(self.N):
            satisfy = self._satisft_KKT(i)
            # 只要有一个不满足KKT条件 就返回False
            if not satisfy:
                return False
        return True

    def _select_two_parameters(self):
        # 不满足停机条件的情况下 根据SMO算法挑选两个参数进行优化
        index_list = [i for i in range(self.N)]
        # 外层循环首先遍历所有满足条件alpha_i属于[0,C]的样本点 即在间隔边界上的支持向量点
        i1_list_1 = filter(lambda i:self.alpha[i]>0 and self.alpha[i] < self.C, index_list) # 选取alpha_i在[0,C]中的
        i1_list_2 = list(set(index_list) - set(i1_list_1)) # 除了在间隔边界上的样本点外的点

        i1_list = list(i1_list_1)
        i1_list.extend(i1_list_2)

        # 外层循环
        for i in i1_list:
            if self._satisft_KKT(i):
                continue # 如果满足KKT条件 则直接跳过 到下一次循环
            E1 = self.E[i]
            max_ = (0,0) # 和找出的第一个变量相对应的第二个变量

            for j in index_list:
                if i==j:
                    continue # 如果i=j 直接跳过
                E2 = self.E[j]
                if abs(E1-E2)>max_[0]:
                    max_ = (abs(E1-E2), j)
        return i, max_[1]


    def train(self, data, label):
        # 初始化LSVM的参数
        self._init_parameters(input_data=data, input_label=label)
        
        # 训练过程
        for times in range(self.Max_Interation):
            logging.debug('Iterater {}'.format(times)) # 记录迭代次数
            print(accuracy_score(test_labels, self.predict(test_features)))
            
            i1,i2 = self._select_two_parameters() # 寻找两个需要优化的变量 算法第二步
            
            # 优化的上下界
            if self.Y[i1] == self.Y[i2]: # 同label
                L = max(0, self.alpha[i2] + self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] + self.alpha[i1])
            else: # 不同label
                L = max(0, self.alpha[i2]-self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2]-self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            eta = self._dot_(self.X[i1], self.X[i1]) + self._dot_(self.X[i2], self.X[i2]) - 2 * self._dot_(self.X[i1], self.X[i2]) 

            alpha2_new_unc = self.alpha[i2] + self.Y[i2]*(E1-E2)/eta
            # 公式(7.108) 李航《统计机器学习》
            alph2_new = 0
            if alpha2_new_unc > H:
                alph2_new = H
            elif alpha2_new_unc < L:
                alph2_new = L
            else:
                alph2_new = alpha2_new_unc

            # 公式(7.109)
            alph1_new = self.alpha[i1] + self.Y[i1] * \
                self.Y[i2] * (self.alpha[i2] - alph2_new)

            # 公式(7.115) 及 公式(7.116)
            b_new = 0
            b1_new = -E1 - self.Y[i1] * self._dot_(self.X[i1], self.X[i1]) * (alph1_new - self.alpha[i1]) - self.Y[i2] * self._dot_(self.X[i2], self.X[i1]) * (alph2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self._dot_(self.X[i1], self.X[i2]) * (alph1_new - self.alpha[i1]) - self.Y[i2] * self._dot_(self.X[i2], self.X[i2]) * (alph2_new - self.alpha[i2]) + self.b

            if alph1_new > 0 and alph1_new < self.C:
                b_new = b1_new
            elif alph2_new > 0 and alph2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            self.alpha[i1] = alph1_new
            self.alpha[i2] = alph2_new
            self.b = b_new

            self.E[i1] = self._E_(i1)
            self.E[i2] = self._E_(i2)

    def _predict_(self,image):
        # 用来单张测试
        result = self.b
        for i in range(self.N):
            result += self.alpha[i]*self.Y[i]*self._dot_(image, self.X[i])
        
        if result > 0:
            return 1 # 标签为正
        else:
            return -1 # 标签为负

    def predict(self,features):
        result = []
        for feature in features:
            result.append(self._predict_(feature))
        return result

if __name__ == "__main__":
    import time
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import accuracy_score
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 使用MNIST进行训练
    from mnist import load_binary_2
    data_dic = load_binary_2(0, 1)
    train_features = data_dic['train_data']
    train_labels = data_dic['train_label']
    test_features = data_dic['test_data']
    test_labels = data_dic['test_label']
    train_features = train_features.reshape(len(train_features), -1)
    test_features = test_features.reshape(len(test_features), -1)
    print('Start read data')

    # 在伪造数据集上进行测试
    start_time = time.time()
    # train_features, train_labels, test_features, test_labels = generate_dataset(2000,visualization=False)
    time_2 = time.time()
    print('read data cost {}'.format(time_2-start_time))
    
    print('Start training')
    svm = LSVM()
    svm.train(train_features, train_labels)
    
    time_3 = time.time()
    print('training cost {}'.format(time_3 - time_2))
    
    print('Start Test')
    test_predict = svm.predict(test_features)
    time_4 = time.time()

    print('predicting cost {}'.format(time_4-time_3))

    score = accuracy_score(test_labels, test_predict)

    print('svm the accuracy is {}'.format(score))