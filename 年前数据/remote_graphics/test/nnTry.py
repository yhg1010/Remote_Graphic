from doctest import OutputChecker
import pandas as pd
import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

input = pd.read_csv("bandsValueTable.csv")
output = pd.read_csv("dan_res.csv")

xTrain, xTest, yTrain, yTest = train_test_split(input, output, test_size = 0.25, random_state = 0)
random.seed(0)

class bpNN() :
    def __init__(self) :
        self.inputNum = 0
        self.hiddenNum = 0
        self.outputNum = 0
        self.inputCells = []
        self.hiddenCells = []
        self.outputCells = []
        self.inputWeights = []
        self.outputWeights = []
        self.inputCorrection = []
        self.outputCorrection = []
    
    def weigthsInit(self, n, m) :
        wei = []
        for i in range(n) :
            wei.append([0.0] * m)
        return wei
    
    def rand_(self, a, b):
        return (b - a) * random.random() + a

    def sigmoid(self, x) :
        if(x < -70):
            x = -70
        return 1.0/(1.0 + math.exp(-x))

    def sigmoidD(self, x) :
        return x * (1 - x)

    def setup(self, inputNum, hiddenNum, outputNum) :
        self.inputNum = inputNum + 1
        self.hiddenNum = hiddenNum
        self.outputNum = outputNum

        self.inputCells = [1.0] * self.inputNum
        self.hiddenCells = [1.0] * self.hiddenNum
        self.outputCells = [1.0] * self.outputNum
        
        self.inputWeights = self.weigthsInit(self.inputNum, self.hiddenNum)
        self.outputWeights = self.weigthsInit(self.hiddenNum, self.outputNum)

        for i in range(self.inputNum) :
            for h in range(self.outputNum) :
                self.inputWeights[i][h] = self.rand_(-0.2,0.2)
        
        for h in range(self.hiddenNum) :
            for o in range(self.outputNum) :
                self.outputWeights[h][o] = self.rand_(-2.0, 2.0)
        
        self.inputCorrection = self.weigthsInit(self.inputNum, self.hiddenNum)
        self.outputCorrection = self.weigthsInit(self.hiddenNum, self.outputNum)

    def predict(self, inputs) :
        for i in range(self.inputNum - 1) :
            self.inputCells[i] = inputs[i]
        
        for h in range(self.hiddenNum) :
            total = 0.0
            for i in range(self.inputNum) :
                total += self.inputCells[i] * self.inputWeights[i][h]
            self.hiddenCells[h] = self.sigmoid(total)
        
        for o in range(self.outputNum) :
            total = 0.0
            for h in range(self.hiddenNum) :
                total += self.hiddenCells[h] * self.outputWeights[h][o]
            self.outputCells[o] = self.sigmoid(total)
        
        return self.outputCells[:]

    def backPropagate(self, inputs, labels, learn, correct) :
        self.predict(inputs)
        outputDealts = [0.0] * self.outputNum

        for o in range(self.outputNum) :
            error = labels[o] - self.outputCells[o]
            outputDealts[o] = self.sigmoidD(self.outputCells[o]) * error
        
        hiddenDealts = [0.0] * self.hiddenNum

        for h in range(self.hiddenNum) :
            error = 0.0
            for o in range(self.outputNum) :
                error += outputDealts[o] * self.outputWeights[h][o]
            hiddenDealts[h] = self.sigmoidD(self.hiddenCells[h]) * error

        for h in range(self.hiddenNum) :
            for o in range(self.outputNum) :
                change = outputDealts[o] * self.hiddenCells[h]
                self.outputWeights[h][o] += change * learn + correct + self.outputCorrection[h][o]
                self.outputCorrection[h][o] = change
        
        for i in range(self.inputNum) :
            for h in range(self.hiddenNum) :
                change = hiddenDealts[h] * self.inputCells[i]
                self.inputWeights[i][h] += change * learn + correct + self.inputCorrection[i][h]
                self.inputCorrection[i][h] = change
        
        error = 0.0
        for o in range(self.outputNum) :
            error += 0.5 * (self.outputCells[o] - labels[o]) ** 2
        return error

    def train(self, inputs, labels, limit = 10000, learn = 0.05, correct = 0.1) :
        inputs = pd.read_csv("bandsValueTable.csv")
        inputs = np.array(inputs)
        labels = pd.read_csv("dan_res.csv")
        labels = np.array(labels)
        self.setup(12, 10, 1) 
        for index in range(limit) :
            error = 0.0
            for i in range(len(inputs)) :
                input = inputs[i]
                label = labels[i]
                error += self.backPropagate(input, label, learn, correct)
    
    def test(self):
        inputs = pd.read_csv("bandsValueTable.csv")
        inputs = np.array(inputs)
        labels = pd.read_csv("dan_res.csv")
        labels = np.array(labels)
        self.setup(12, 10, 1) #初始化神经网络
        self.train(inputs[:, 3 : 15], labels[:, 3], 100, 0.05, 0.1)
        # plsrR2 = r2_score(labels, self.outputCells)
        # plsrRmse = mean_squared_error(labels, self.outputCells)
        # print("plsrR2 : ", plsrR2)
        # print("plsrRmse : ", np.sqrt(plsrRmse))
        print(self.outputCells)

if __name__ == "__main__" :
    model = bpNN()
    model.test()


