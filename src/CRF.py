
import sys
from matplotlib.pyplot import polar
sys.path.append(".")
import numpy as np
import os
from numpy.lib.npyio import load
from feature_function import StatusFF, ObsFF, TransFF
from utils import load_dataset, LabelDict, pooling
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class TrainArg:
    lr = 0.001
    epochs = 150

class CRF_OCR(object):
    def __init__ (self, d:LabelDict, X_dim):
        self.d = d
        self.X_dim = X_dim
        self.y_dim = len(self.d)

        # 构造特征向量簇
        self.feature_functions, self.ffshape = self._construct_feature_functions()
        
        self.crf = CRF(d, self.ffshape)

        self.optimizer = optim.Adam(self.crf.parameters(), lr=TrainArg.lr, weight_decay=0.01)

        self.global_step = 0
        self._best_acc = 0
        self._best_model = None


    def train(self, train_dataset, test_dataset):
        """
        使用train_dataset对crf的权重向量进行训练，在test_datset上进行测试
        """
        # 使用特征函数簇作用于训练集的输入，将原来的像素点转换为特征向量
        train_dataset = self._convert_dataset(train_dataset)
        test_dataset = self._convert_dataset(test_dataset)

        train_order = np.arange(0, len(train_dataset[0]))
        for i_epoch in range(TrainArg.epochs):
            np.random.shuffle(train_order)
            print("========== Epoch {} ==========".format(i_epoch))
            for i_data in range(len(train_dataset[0])):
                self.global_step += 1

                X, label = train_dataset[0][train_order[i_data]], train_dataset[1][train_order[i_data]]
                label = np.asarray([self.d[i] for i in label])

                # 损失函数为对数似然的负数
                loss = self.crf.neg_log_likelihood(X, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 每200次更新后，在测试集上进行测试
                if self.global_step % 200 == 0:
                    print("Global step {}: ".format(self.global_step), end="")
                    self.test(test_dataset)

        # 训练完成后，使用最好的模型在测试集上进行测试，输出单词准确率和字符预测准确率
        print("Training finished. The best model's performance on test dataset is")
        self.crf = copy.deepcopy(self._best_model)
        self.test(test_dataset)
    
    def _construct_feature_functions(self):
        """
        根据数据的维度和标签的维度，生成特征向量簇。
        为了进行优化，这里生成的实际上是一个特征向量矩阵。
        """
        ffs = []
        for c in range(self.y_dim):
            # 观察特征函数
            tmp = [ObsFF(j, c, 1) for j in range(self.X_dim)]
            # 转移特征函数
            tmp.extend([TransFF(c, d) for d in range(self.y_dim)])
            # 状态特征函数
            tmp.append(StatusFF(c))
            ffs.append(tmp)

        return ffs, (self.y_dim, len(ffs[0]))

    def _convert_dataset(self, dataset):
        """
        使用特征函数簇作用于训练集的输入，将原来的像素点转换为特征向量
        """
        print("Converting dataset. This may take for about 2 min s ...")
        for i in range(len(dataset[0])):
            tmp = dataset[0][i]
            length = len(tmp)

            Ms1 = [[[
                [0]*(self.ffshape[1]*y) + [f(y_, y, tmp, i) for f in self.feature_functions[y]] + [0]*(self.ffshape[1]*(self.ffshape[0]-y-1))
                    for y in range(self.y_dim) ]
                        for y_ in range(self.y_dim) ] 
                            for i in range(length) ]
            Ms1 = torch.FloatTensor(Ms1)
            dataset[0][i] = Ms1
        return dataset

    def test(self, test_dataset):
        """
        在测试集上对当前模型进行测试
        """
        char_acc = 0
        char_total = 0
        word_acc = 0
        word_total = 0
        for i_data in range(len(test_dataset[0])):
            X, label = test_dataset[0][i_data], test_dataset[1][i_data]

            out = [self.d[id] for id in self.crf.test(X)]
            prediction = "".join(out)

            if prediction == label:
                word_acc += 1
            for i in range(len(out)):
                if out[i] == label[i]:
                    char_acc += 1
            word_total += 1
            char_total += len(out)
        print("performance on test dataset: word precision {:.4f}, char precision {:.4f} ".format(word_acc/word_total, char_acc/char_total))
        
        if word_acc > self._best_acc:
            self._best_model = copy.deepcopy(self.crf)
            if not os.path.exists("./state_dicts"):
                os.mkdir("./state_dicts")
            torch.save(self._best_model.state_dict(), "./state_dicts/acc_{:.4f}".format(word_acc/word_total))
            self._best_acc = word_acc



class CRF(nn.Module):
    def __init__ (self, d:LabelDict, ffshape):
        super(CRF, self).__init__()
        self.d = d
        self.L = len(d)
        self.K = 1
        for d in ffshape:
            self.K *= d

        self.w = nn.Parameter(torch.zeros(self.K))
        torch.nn.init.normal(self.w, mean=0, std=1)

    def forward(self, X):
        """
        计算矩阵M_i，对应于《统计学习方法》书中的矩阵。为了防止上溢，这里为对数概率矩阵。
        """
        return X @ self.w

    def neg_log_likelihood(self, X, label):
        """
        计算似然函数的负对数
        """
        alpha = self._forward(X)
        length = len(X)

        logZ = torch.logsumexp(alpha[-1, :], dim=0)

        label_path_score = 0
        label_ = self.d["[BEG]"]
        for i in range(1, length+1):
            tmp = X[i-1, label_, label[i-1]]
            label_path_score += torch.FloatTensor(tmp) @ self.w
            label_ = label[i-1]
        
        return -label_path_score + logZ

    def _forward(self, X):
        """
        根据crf计算出的对数概率矩阵，计算前向概率alpha
        """
        Ms = self.forward(X)

        BEG = self.d["[BEG]"]
        length = len(X)

        alpha = torch.zeros(length+1, self.L)
        alpha[1, :] = Ms[0, BEG, :]
        for i in range(2, length+1):
            alpha[i, :] = torch.logsumexp(alpha[i-1, :].view(self.L, -1) + Ms[i-1], dim=0)
        return alpha

    def test(self, X):
        """
        使用viterbi算法，得到最可能的标记序列
        """
        length = len(X)
        Ms = self.forward(X)
        BEG = self.d["[BEG]"]

        with torch.no_grad():
            viterbi = torch.zeros(length, self.L)
            note = torch.zeros(length, self.L)
            viterbi[0, :] = Ms[0, BEG, :]
            note[0, :] = BEG
            # viterbi算法的前向过程
            for i in range(1, length):
                viterbi[i, :], note[i, :] = torch.max(viterbi[i-1, :].view(self.L, -1) + Ms[i], dim=0)
            # 基于note备忘录进行回溯
            label = []
            last = viterbi[-1, :].argmax().item()
            for i in range(length-1, -1, -1):
                label.insert(0, last)
                last = note[i, last].long().item()
        return label
        

if __name__ == "__main__":

    train_dataset = load_dataset("./Dataset/train")
    test_dataset = load_dataset("./Dataset/test")
    # train_dataset = pooling(train_dataset)
    # test_dataset = pooling(test_dataset)
    label_dict = LabelDict(train_dataset[1]+test_dataset[1])

    crf_ocr = CRF_OCR(label_dict, 321)
    crf_ocr.train(train_dataset, test_dataset)
