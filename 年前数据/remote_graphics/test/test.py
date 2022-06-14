from osgeo import gdal
import os
import numpy as np

class GRID:
    def __init__(self, filename):
        self.file = filename
        
    #读图像文件
    def read_img(self):
        dataset = gdal.Open(self.file)       

        im_width = dataset.RasterXSize    #栅格矩阵的列数
        im_height = dataset.RasterYSize   #栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
        im_proj = dataset.GetProjection() #地图投影信息
        #print(im_width)
        #print("------")
        #print(im_height)
        #print("------")
        im_data = dataset.ReadAsArray(xoff = 0, yoff = 0, xsize = im_width, ysize = im_height) #将数据写成数组，对应栅格矩阵
        im_bands = dataset.RasterCount	
        
        del dataset 
        return im_proj, im_geotrans, im_bands
    
    def read_bands(self):
        dataset = gdal.Open(self.file)

        band1 = dataset.GetRasterBand(1)
        band2 = dataset.GetRasterBand(2)
        band3 = dataset.GetRasterBand(3)
        band4 = dataset.GetRasterBand(4)

        im_datas1 = band1.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
        im_datas1 = im_datas1.astype(np.float)

        im_datas2 = band2.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
        im_datas2 = im_datas2.astype(np.float)

        im_datas3 = band3.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
        im_datas3 = im_datas3.astype(np.float)

        im_datas4 = band4.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
        im_datas4 = im_datas4.astype(np.float)

        print(im_datas1[10, 10])
        
        del dataset

if __name__ == "__main__":
    os.chdir(r'../data/preprocess/gaofen/10.8/')                        #切换路径到待处理图像所在文件夹
    #run = GRID(r'GF1B_PMS_E123.4_N47.9_20211012_L1A1228046777-MUX_apprad_FLAASH_fuse.tif')
    
    run = GRID(r'GF1B_PMS_E122.8_N47.9_20211008_L1A1228044962-MUX_apprad_FLAASH_fuse.tif')
    proj, geotrans, bandsCount = run.read_img()        #读数据
    print(proj)
    print("--------")
    print(geotrans)
    print("--------")
    print(bandsCount)
    print("--------")
    #run.read_bands()
#    print(data.shape)
    #run.write_img('LC81230402013164LGN00_Rewrite.tif',proj,geotrans,data) #写数据
