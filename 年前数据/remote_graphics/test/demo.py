from osgeo import gdal
import numpy as np
import os
import matplotlib.pyplot as plt

filename = '/home/remote/data/preprocess/gaofen/10.12/GF1B_PMS_E123.4_N47.9_20211012_L1A1228046777-MUX_apprad_FLAASH_fuse.tif'

dataset = gdal.Open(filename, gdal.GA_ReadOnly)
if not dataset:
    print('error!')
    exit(-1)



print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                            dataset.GetDriver().LongName))
print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                    dataset.RasterYSize,
                                    dataset.RasterCount))
print("Projection is {}".format(dataset.GetProjection()))

geotransform = dataset.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

band = dataset.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min,max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min,max))

if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))

if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

scanline = band.ReadRaster(xoff=0, yoff=0,
                        xsize=band.XSize, ysize=1,
                        buf_xsize=band.XSize, buf_ysize=1,
                        buf_type=gdal.GDT_Float32)

im = dataset.ReadAsArray(dataset.RasterXSize//2, dataset.RasterYSize//2, dataset.RasterXSize//8, dataset.RasterYSize//8)
print(np.max(im), np.min(im))
im = np.transpose(im, [1, 2, 0])

plt.imsave('im.jpg', im[:, :, :3])