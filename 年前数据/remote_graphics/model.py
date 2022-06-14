#python model.py dan_res.csv lin_res.csv jia_res.csv bandsValueTableSB2.csv
#from statistics import linear_regression
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cross_decomposition import PLSRegression
import sys
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
#bandsValueTable`size : 109*15. col1 is correspond to lon and col2 is correspond to lat. col3-col14 are the bands value. col15 is the index of the sample points 
#dan_res.csv lin_res.csv jia_res.csv are the label data extract from the soil_data.xlsx

class RemoteGraphicsReg :
    #传入氮磷钾的csv文件
    def __init__(self, filename1, filename2, filename3, filename4) :
        self.dan = filename1
        self.lin = filename2
        self.jia = filename3
        self.danLabelData = pd.read_csv(self.dan) 
        self.danLabelData = np.array(self.danLabelData)
        self.linLabelData = pd.read_csv(self.lin)
        self.linLabelData = np.array(self.linLabelData)
        self.jiaLabelData = pd.read_csv(self.jia)
        self.jiaLabelData = np.array(self.jiaLabelData)
        self.bands = pd.read_csv(filename4)
        self.bands = np.array(self.bands)

    #预处理    
    def preProcess(self) :
        #self.danLabelData = self.danLabelData.dropna()
        #self.linLabelData = self.linLabelData.dropna()
        #self.jiaLabelData = self.jiaLabelData.dropna()
        print(self.bands[:, 3 : 15])
        print(self.danLabelData[:, 3])

    #准备训练测试数据
    def dataPrepare(self, flag) :
        if(flag == 0) :
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.bands[:, 3 : 15], self.danLabelData[:, 3], test_size = 0.25, random_state = 1)
        elif(flag == 1) :
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.bands[:, 3 : 15], self.linLabelData[:, 3], test_size = 0.25, random_state = 1)
        else :
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.bands[:, 3 : 15], self.jiaLabelData[:, 3], test_size = 0.25, random_state = 1)
        self.x_train = self.x_train
        self.x_test = self.x_test
    #回归模型，参数
    def plsrModel(self, bandsNum) :
        plsrModel = PLSRegression(scale = True)
        paraGrid = {'n_components' : range(1, bandsNum)}

        gridSearch = GridSearchCV(plsrModel, paraGrid)
        plsrModelRes = gridSearch.fit(self.x_train, self.y_train)

        y_pre = plsrModelRes.predict(self.x_test)
        plsrR2 = r2_score(self.y_test, y_pre)
        plsrRmse = mean_squared_error(self.y_test, y_pre)
        chosenBands = plsrModelRes.best_params_
        print("plsrR2 : ", plsrR2)
        #print("plsrRmse : ", np.sqrt(plsrRmse))
        #print("best params of the plsrModelRes", chosenBands)
    
    def svrModel(self) :
        svr = SVR(kernel = "sigmoid")
        svr.fit(self.x_train, self.y_train)
        y_pre = svr.predict(self.x_test)
        svrR2 = r2_score(self.y_test, y_pre)
        print("svrR2 : ", svrR2)
    
    def linRegression(self) :
        linReg = LinearRegression()

        linReg.fit(self.x_train, self.y_train)
        y_pred = linReg.predict(self.x_test)
        linRegR2 = r2_score(self.y_test, y_pred)
        print("linReg R2 score : ", linRegR2)

if __name__ == "__main__" :
    model = RemoteGraphicsReg(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    model.dataPrepare(0)
    #model.preProcess()
    model.plsrModel(12)
    model.linRegression()
    model.svrModel()

