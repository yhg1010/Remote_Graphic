#python dan_lin_jiaProcess.py "速效钾" jia.csv jia_res.csv
import pandas as pd
import numpy as np
import sys

class dataProcess() :

    def __init__(self, sheetname, outputCsv, outputCsvRes) :
        self.sheetname = sheetname
        self.ouputCsv = outputCsv
        self.outputCsvRes = outputCsvRes

    def process(self) :
        data = pd.read_excel("soil_data.xlsx", self.sheetname)
        data = np.array(data)
        #print(data)
        #print(data[2][4])
        dataSave = np.empty(shape = [0, 2])
        for i in range(2, 12) :
            dataSave = np.append(dataSave, [[data[i][0], data[i][4]]], axis = 0)

        for i in range(13, 188) :
            dataSave = np.append(dataSave, [[data[i][0], data[i][4]]], axis = 0)

        dataSave = pd.DataFrame(dataSave)
        dataSave.to_csv(self.ouputCsv)

        data1 = pd.read_csv(self.ouputCsv)
        data1 = np.array(data1)
        data2 = pd.read_csv("bandsValueTableSB2.csv")
        data2 = np.array(data2)
        i = 0 
        j = 0
        while(i < 163 and j < 163) :
            if(int(data1[i][1]) == int(data2[j][15])) :
                i = i + 1
                j = j + 1
            else :
                print(i)
                data1 = np.delete(data1, i, 0)
        for a in range(19) :
            data1 = np.delete(data1, i, 0)

        data1 = pd.DataFrame(data1)
        data1.to_csv(self.outputCsvRes)

        print(data1)

        # print(data1[0][1])
        # print(data2[0][15])

if __name__ == "__main__" :
    model = dataProcess(sys.argv[1], sys.argv[2], sys.argv[3])
    model.process()