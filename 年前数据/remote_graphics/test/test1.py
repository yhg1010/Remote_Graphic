# import numpy as np

# class Test :
#     def __init__(self) :
#         self.num = 1
#         self.a = 2

#     def add(self) :
#         self.addres = self.a + self.num

#     def print(self) :
#         print(self.addres)
    
#     def arrayTest(self) :
#         res = np.empty(shape = [0, 2])
        
#         for i in range(10) :
#             tem = [i, i+1]
#             #tem.insert(i + 1)
#             np.insert(res, 0, tem, axis = 0)
#         print(res)
    
# test = Test()
# test.add()
# test.print()
# test.arrayTest()


# import numpy as np

# n = 2
# X = np.empty(shape=[0, n])

# for i in range(5):
#     for j in range(2):
#         X = np.append(X, [[i, j]], axis=0)

# print(X)

import pandas as pd
import numpy as np
from osgeo import gdal
import os
import osr
import matplotlib.pyplot as plt
import scipy.optimize as optimize
#df = pd.read_excel('data/soil_data.xlsx', "坐标")
#dataArray = np.array(df)
#print(dataArray)

def coordTransf(Xpixel, Ypixel):
    os.chdir('/home/remote/data/preprocess/sentinel/5.1')
    dataset = gdal.Open('Us2UDHWU2021-04-23_2021-05-01U_cp.tif')
    GeoTransform = dataset.GetGeoTransform()
    XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(XGeo, YGeo) #coords[0]是经度,coords[1]是纬度
    return coords[0], coords[1]

def decoordTransf(xGeo, yGeo) :
    os.chdir('/home/remote/data/preprocess/sentinel/5.1')
    dataset = gdal.Open('Us2UDHWU2021-04-23_2021-05-01U_cp.tif')
    GeoTransform = dataset.GetGeoTransform()
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    ct2 = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct2.TransformPoint(xGeo, yGeo)
    proLon = coords[0]
    proLat = coords[1]
    xPixel = int((proLon - GeoTransform[0]) / GeoTransform[1] + 0.000000001)
    yPixel = int((GeoTransform[3] - proLat) / -GeoTransform[5] + 0.000000001)
    #xPixel*GeoTransform[1] = xGeo*GeoTransform[4] - GeoTransform[0]*GeoTransform[4] - yPixel * GeoTransform[2]*GeoTransform[4] 
    #xPixel*GeoTransform[4] = yGeo*GeoTransform[1] - GeoTransform[3]*GeoTransform[1] - yPixel * GeoTransform[5]*GeoTransform[1]
    #yPixel = (xGeo * GeoTransform[4] - GeoTransform[0] * GeoTransform[4] -yGeo * GeoTransform[1] + GeoTransform[3] * GeoTransform[1])/(GeoTransform[2] * GeoTransform[4] - GeoTransform[1] * GeoTransform[5])
    #xPixel = (xGeo - GeoTransform[0] - yPixel * GeoTransform[2])/GeoTransform[1]
    return xPixel, yPixel

# xGeoTrue = dataArray[1][2]
# yGeoTrue = dataArray[1][1]
# xPixel, yPixel = decoordTransf(xGeoTrue, yGeoTrue)
# print("xGeoTrue : ", xGeoTrue, "yGeoTrue : ", yGeoTrue)
# xGeoPred, yGeoPred = coordTransf(xPixel, yPixel)
# print("xGeoPred : ", xGeoPred, "yGeoPred : ", yGeoPred)

# for i in range(184) :
#     yGeo = dataArray[i][1]
#     xGeo = dataArray[i][2]
#     print(i)
#     print(xGeo, yGeo)
# #     print(decoordTransf(xGeo, yGeo))

# os.chdir('/home/remote/data/preprocess/sentinel/5.1')
# dataset = gdal.Open('Us2UDHWU2021-04-23_2021-05-01U_cp.tif')
# band1 = dataset.GetRasterBand(1)
# lon, lat = coordTransf(1532, 1820)
# print("geo at 1532,1820 : ", lon, lat)
# print("pixel at 123.1909,47.6972 : ", decoordTransf(123.1909, 47.6972))

datay = pd.read_csv("data/dan_res.csv")
datax = pd.read_csv("data/bandsValueTableSB2.csv")
datax = np.array(datax)
datay = np.array(datay)


def target_func(x, a0, a1, a2):
    return a0 * np.exp(-x / a1) + a2

fig, ax = plt.subplots()

x = datax[:, 6]
y = datay[:, 3]
a0 = max(y) - min(y)
a1 = x[round(len(x) / 2)]
a2 = min(y)
p0 = [a0, a1, a2]
print(p0)
para, cov = optimize.curve_fit(target_func, x, y, p0=p0)
print(para)
y_fit = [target_func(a, *para) for a in x]
ax.plot(x, y_fit, 'g')

plt.show()




for i in range(3, 15) :
    x = datax[:, i]
    y = datay[:, 3]

    print(np.corrcoef(x, y))
#print(datax[:, 3 : 15])
#print(datay[:, 3])

