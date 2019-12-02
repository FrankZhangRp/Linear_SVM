import numpy as np
def load_train():
    images = np.load(r'G:\Machine_Learning_LiHang\data\mnist_npy\train_images.npy')
    labels = np.load(r'G:\Machine_Learning_LiHang\data\mnist_npy\train_labels.npy')
    return {'images': images, 'labels': labels}

def load_test():
    images = np.load(r'G:\Machine_Learning_LiHang\data\mnist_npy\test_images.npy')
    labels = np.load(r'G:\Machine_Learning_LiHang\data\mnist_npy\test_labels.npy')
    return {'images': images, 'labels': labels}
def load_number(num, total_data, total_label):
    num_size = sum(total_label == num)
    num_data = np.zeros(shape=[num_size, 28,28])
    num_label = np.ones(shape=[num_size]) * num
    index_num = 0
    for i, flag in enumerate(list(total_label==num)):
        if flag:
            num_data[index_num] = total_data[i]
            index_num += 1
    return num_data, num_label

def load_binary_1(num1, num2):
    train_dic = load_train()
    test_dic = load_test()
    num1_train_data, num1_train_label = load_number(num1, train_dic['images'], train_dic['labels'])
    num2_train_data, num2_train_label = load_number(num2, train_dic['images'], train_dic['labels'])
    num1_test_data, num1_test_label = load_number(num1, test_dic['images'], test_dic['labels'])
    num2_test_data, num2_test_label = load_number(num2, test_dic['images'], test_dic['labels'])
    out_dic = {'num1':num1,
    'num2': num2,
    'num1_train_data': num1_train_data,
    'num1_train_label': num1_train_label,
    'num1_test_data': num1_test_data,
    'num1_test_label': num1_test_label,
    'num2_train_data': num2_train_data,
    'num2_train_label': num2_train_label,
    'num2_test_data': num2_test_data,
    'num2_test_label': num2_test_label
    }
    return out_dic


def load_binary_2(num1, num2):
    train_dic = load_train()
    test_dic = load_test()
    num1_train_data, num1_train_label = load_number(num1, train_dic['images'], train_dic['labels'])
    num2_train_data, num2_train_label = load_number(num2, train_dic['images'], train_dic['labels'])
    num1_test_data, num1_test_label = load_number(num1, test_dic['images'], test_dic['labels'])
    num2_test_data, num2_test_label = load_number(num2, test_dic['images'], test_dic['labels'])
    num1_train_label = np.ones(shape=num1_train_label.shape)
    num1_test_label = np.ones(shape=num1_test_label.shape)
    num2_train_label = np.ones(shape=num2_train_label.shape) * -1
    num2_test_label = np.ones(shape=num2_test_label.shape) * -1
    out_dic = {
    'train_data': np.concatenate((num1_train_data, num2_train_data), axis=0),
    'train_label': np.concatenate((num1_train_label, num2_train_label), axis=0),
    'test_data': np.concatenate((num1_test_data, num2_test_data), axis=0),
    'test_label': np.concatenate((num1_test_label, num2_test_label), axis=0),
    }
    return out_dic

if __name__ == '__main__':
    train_dic = load_train()
    test_dic = load_test()
    one_data, one_label = load_number(1, train_dic['images'], train_dic['labels'])