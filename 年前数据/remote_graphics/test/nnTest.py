import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split

input =pd.read_excel("C:/Users/Lenovo/Desktop/数据集/in.xlsx")
output= pd.read_excel("C:/Users/Lenovo/Desktop/数据集/out.xlsx")
x_train, x_test, y_train, y_test = train_test_split(input,output,test_size=0.2, random_state=0)#导入数据
random.seed(0)
#在数区间 a ~ b 中，随机生成一个float数
def rand(a, b):
    return (b - a) * random.random() + a# random.random() 生成一个0~1的浮点数
#创建一个指定大小的 矩阵，并用fill 去填充
def make_matrix(m, n, fill = 0.0 ):
    mat = []
    for i in range(m):# 对行进行循环
        mat.append([fill] * n)#创建每行的列元素
    return mat
#定义sigmoid 函数，及它的导数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
def sigmoid_derivate(x):
    return x * (1 - x)
#定义BPNeuralNetwork类，使用三个列表维护输入层，隐含层，输出层神经元
class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []
#定义setup 的方法初始化神经网络
    # ni ,nh ,no ->各层神经元的个数
    def setup(self, ni, nh, no):
        self.input_n = ni + 1 #输入层额外加一个偏置神经元，提供一个可控的输入修正；（或者为每个隐含层神经元设置一个偏置参数）
        self.hidden_n = nh  #隐藏层神经元个数
        self.output_n = no  #输出层神经元个数
        #init cells
        #初始化神经元的输出值
        self.input_cells = [1.0] * self.input_n #输入层各神经元的值初始化为1
        self.hidden_cells = [1.0] * self.hidden_n#隐藏层神经元的值初始化为1
        self.output_cells = [1.0] * self.output_n#输出层神经元的值初始化为1
        # init weights
        #初始化神经网络各层权重的值 各层的权重已矩阵的形式存储
        self.input_weights = make_matrix(self.input_n, self.hidden_n) #初始化输入层与隐藏层间的权重 
        self.output_weights = make_matrix(self.hidden_n,self.output_n)#初始化隐藏层与输出层间的权重
        #random activate
        #给权重矩阵 随机赋初值
        for i in range(self.input_n):#给输入层及隐藏层间的权重矩阵赋初值
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):#给隐藏层及输出层间的权重赋初值
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        #init correction matrix
        #创建矫正矩阵 此处应该是指各层权重矩阵的矫正矩阵
        self.input_correction = make_matrix(self.input_n, self.hidden_n) #输入矫正矩阵
        self.output_correction = make_matrix(self.hidden_n, self.output_n)#输出矫正矩阵
         #定义predict方法进行一次前馈，并返回输出
    def predict(self,inputs):
        #activate input layer
        #激活输入层
         for i in range(self.input_n - 1):
             self.input_cells[i] = inputs[i] #对输入层神经元赋值 （此处不含输入层神经元的偏置）
         #activate hidden layer
         #激活隐藏层
         for j in range(self.hidden_n):#对隐含层神经元进行计算求值
             total = 0.0
             for i in range(self.input_n):#前一层神经元的输出 * 相应权重值 后求和
                 total += self.input_cells[i] * self.input_weights[i][j] #
             self.hidden_cells[j] = sigmoid(total)#对每个神经元经过前一层的求和后 计算经过所选激励函数映射后的输出。
          #activate output layer
          #激活输出层
         for k in range(self.output_n):#对输出层各神经元的值进行计算
             total = 0.0
             for j in range(self.hidden_n):#隐藏层的输出经过神经元的加权后求和
                 total += self.hidden_cells[j] * self.output_weights[j][k]
             self.output_cells[k] = sigmoid(total)#所得和经过激励函数的映射后的输出
         return self.output_cells[:]
#定义一次反向传播 和更新权值的过程，并返回最终预测误差
    def back_propagate(self, case, label, learn, correct):
        #feed forward ->前馈
        self.predict(case) #对实例case进行前馈预测
        #get output layer error ->获取输出层误差 
        output_deltas = [0.0] * self.output_n #初始化输出层更新差值列表
        #计算实际的标签与预测标签的差值进行计算
        for o in range(self.output_n):
            error = label[o] -self.output_cells[o]
            output_deltas[o] = sigmoid_derivate(self.output_cells[o]) * error#？？？
        #get hidden layer error ->获取隐藏层的误差列表
        hidden_deltas = [0.0] * self.hidden_n #初始化隐藏层更新差值列表
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivate(self.hidden_cells[h]) * error
        # update output weights 更新输出层权重
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct + self.output_correction[h][o]
                self.output_correction[h][o] = change
        #update input weights    更新输入层权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct *self.input_correction[i][h] #correct 为矫正矩阵的矫正率
                self.input_correction[i][h] = change #更新矫正矩阵
        #get global error 获取全局误差 
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error
  #定义train方法控制迭代，该方法可以修改最大迭代次数，学习率 矫正率 三个参数
    def train(self, cases, labels, limit = 10000,learn = 0.05 , correct = 0.1):
        for i in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
    #test 方法演示如何使用神经网络学习异或逻辑            
    def test(self):
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
                 ]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 5, 1) #初始化神经网络
        self.train(cases, labels, 10000, 0.05, 0.1)#神经网络的学习
        for case in cases: #对输入进行分类预测
            print(self.predict(case))
            
if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()
 


