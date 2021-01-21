import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import allclose
from numpy.lib.npyio import load

def load_dataset(root_path, plot=False):
    """
    从指定目录下读取文件内容
    """
    X = []
    y = []
    for filename in os.listdir(root_path):
        with open (os.path.join(root_path, filename)) as fp:
            label = fp.readline().strip().lower()
            data = [i.strip().split(" ") for i in fp.readlines()]
            data = np.asarray(data, dtype=int)
        X.append(data)
        y.append(label)

    if plot:
        sample_label = list(y[0])
        length = len(sample_label)
        sample_X = X[0]
        _, axes = plt.subplots(3, 3)
        for i in range(min(length, 9)):
            axes[i//3, i%3].imshow(np.transpose(sample_X[i, 1:].reshape(16, 20)))
            axes[i//3, i%3].set_title(sample_label[i])
            axes[i//3, i%3].set_xticks([])
            axes[i//3, i%3].set_yticks([])
        plt.show()
    return X, y

class LabelDict(object):
    """
    标注（字母）和数字id之间相互转换的字典
    """
    def __init__ (self, labels):
        self.l2i = dict()
        self.i2l = dict()
        alphabet = set()
        for label_set in labels:
            for label in label_set:
                for char in label:
                    alphabet.add(char)
        alphabet = list(alphabet)
        self.l2i = dict(zip(alphabet, range(1, len(alphabet)+1)))
        self.i2l = dict(zip(range(1, len(alphabet)+1), alphabet))
        self.l2i["[BEG]"] = 0
        self.i2l["[BEG]"] = 1

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.l2i.get(key, 0)
        elif isinstance(key, int):
            return self.i2l.get(key, "[UNK]")
        else:
            raise TypeError

    def __len__ (self):
        return len(self.l2i)

def pooling(dataset):
    """
    对原本的图像作stride=2的平均池化，用来降低噪点的影响
    """
    for i in range(len(dataset[0])):
        tmp = dataset[0][i][:, 1:].reshape(dataset[0][i].shape[0], 20, 16)
        new = (tmp[:, :-1, :-1] + tmp[:, 1:, :-1]+tmp[:, :-1, 1:]+tmp[:, 1:, 1:])/4
        dataset[0][i][:, 1:] = tmp.reshape(tmp.shape[0], -1)
    return dataset


if __name__=="__main__":
    X_train, y_train = load_dataset("./Dataset/train", plot=True)
    X_test, y_test = load_dataset("./Dataset/test", plot=True)
    
    label_dict = LabelDict([y_train, y_test])

    a = [len(i) for i in y_train]
    b = [len(i) for i in y_test]
