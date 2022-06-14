#python dataPrepare.py /home/remote/data/preprocess/sentinel/5.1 Us2UDHWU2021-04-23_2021-05-01U_cp.tif
from turtle import shape
import numpy as np
from osgeo import gdal
import pandas as pd
import os
import sys
import osr

#from test1 import coordTransf
class dataProcess :
    def __init__(self, dirname, filename) :
        self.dir = dirname
        self.file = filename
        self.dataset = gdal.Open(self.file)

    #pixelCoord trans to geoCoord
    def coordTransf(self, Xpixel, Ypixel):
        GeoTransform = self.dataset.GetGeoTransform()
        XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
        YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(self.dataset.GetProjection())
        geosrs = prosrs.CloneGeogCS()
        ct = osr.CoordinateTransformation(prosrs, geosrs)
        coords = ct.TransformPoint(XGeo, YGeo) #coords[0]是经度,coords[1]是纬度
        return coords[0], coords[1]

    #geoCoord trans to pixelCoord
    def decoordTransf(self, xGeo, yGeo) :
        GeoTransform = self.dataset.GetGeoTransform()
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(self.dataset.GetProjection())
        geosrs = prosrs.CloneGeogCS()
        ct2 = osr.CoordinateTransformation(geosrs, prosrs)
        coords = ct2.TransformPoint(xGeo, yGeo)
        proLon = coords[0]
        proLat = coords[1]
        xPixel = int((proLon - GeoTransform[0]) / GeoTransform[1] + 0.000000001)
        yPixel = int((GeoTransform[3] - proLat) / -GeoTransform[5] + 0.000000001)
        return xPixel, yPixel

    #remote_graphics`s data preprocess
    def remoteDataProcess(self) :
        print("bands count : ", self.dataset.RasterCount)
        band1 = self.dataset.GetRasterBand(1)
        band1Data = band1.ReadAsArray(0, 0, band1.XSize, band1.YSize)

        print("band1`s value at 0,0", band1Data[0][0])
        print("band1Data`s type", type(band1Data))
        print("band1`s xsize : ", band1.XSize, "band1`s ysize : ", band1.YSize)
        print("band1`s max value : ", np.max(band1Data), "band1`s min value : ", np.min(band1Data)) 
        #np.savetxt("bands_value.csv", band1Data, delimiter = ',')
        #band1Data.tofile('bands_value.csv',sep=',',format='%10.5f')
        #print(band1Data.head())

        # band1Data = pd.DataFrame(band1Data)
        # print(band1Data)
        # band1Data.to_csv('bands_value.csv')


    #extract the soildata from the execl file
    def genCoordTransTable(self) :
        soilData = pd.read_excel("data/soil_data.xlsx", "坐标")
        soilData = np.array(soilData)
    
        dataSave = np.empty(shape = [0, 5])
        for i in range(1, 11) :
            xPixel, yPixel = self.decoordTransf(soilData[i][2], soilData[i][1])
            xPixel = int(xPixel)
            yPixel = int(yPixel)
            dataSave = np.append(dataSave, [[soilData[i][2], soilData[i][1], xPixel, yPixel, i]], axis = 0)
        for i in range(12, 187) :
            xPixel, yPixel = self.decoordTransf(soilData[i][2], soilData[i][1])
            xPixel = int(xPixel)
            yPixel = int(yPixel)
            dataSave = np.append(dataSave, [[soilData[i][2], soilData[i][1], xPixel, yPixel, i - 11]], axis = 0)
        
        dataSave = pd.DataFrame(dataSave)
        dataSave.to_csv('coordTransTable.csv')
    
    #extract bands value from the dataset
    def genBandsValueTable(self) :
        coordTransf = pd.read_csv('coordTransTable.csv')
        coordTransf = np.array(coordTransf)
        #print(coordTransf[0][0], coordTransf[0][1], coordTransf[0][2], coordTransf[0][3])
        dataSave = np.empty(shape = [0, 15])
        for i in range(184) :
            xPixel = coordTransf[i][3]
            yPixel = coordTransf[i][4]
            if(xPixel >= 0 and xPixel < self.dataset.RasterXSize and yPixel >= 0 and yPixel < self.dataset.RasterYSize) :
                tem = []
                tem = np.append(tem, xPixel)
                tem = np.append(tem, yPixel)
                for bandIndex in range(1, 13) :
                    bandTem = self.dataset.GetRasterBand(bandIndex)
                    bandTem = bandTem.ReadAsArray(0, 0, bandTem.XSize, bandTem.YSize)
                    xPixel = int(xPixel)
                    yPixel = int(yPixel)
                    tem = np.append(tem, bandTem[yPixel][xPixel])
                print(coordTransf[i][5])
                tem = np.append(tem, coordTransf[i][5])
                dataSave = np.append(dataSave, [tem], axis = 0)
                
        dataSave = pd.DataFrame(dataSave)
        dataSave.to_csv('bandsValueTable.csv')
                

if __name__ == "__main__" :
    os.chdir(sys.argv[1])
    obj = dataProcess(sys.argv[1], sys.argv[2])
    #obj.remoteDataProcess()
    obj.genCoordTransTable()
    #obj.remoteDataProcess()
    obj.genBandsValueTable()