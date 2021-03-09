import os
import sys
import gdal

def split_tiles(path, file_list, tile_size):
    
    count=0
    for url in file_list:
        os.system("cd {} && wget {}".format(path, url))
        
        file = url.split('/')[-1]

        output_filename = 'tile_{}_'.format(count)

        tile_size_x, tile_size_y = tile_size, tile_size

        ds = gdal.Open(path + file)
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize

        for i in range(0, xsize, tile_size_x):
            for j in range(0, ysize, tile_size_y):
                com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(path) + str(file) + " " + str(path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
                os.system(com_string)
        
        os.system("rm {}{}".format(path, file))
        count+=1 
        print("\nCOMPLETED {} OF {} FILES\n".format(count, len(file_list)))
             
        
file_list = [
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000000000-0000000000.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000000000-0000023296.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000000000-0000046592.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000023296-0000000000.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000023296-0000023296.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000023296-0000046592.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000046592-0000000000.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000046592-0000023296.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000046592-0000046592.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000069888-0000000000.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000069888-0000023296.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/18Q/S2_percentile_30_UTM_437-0000069888-0000046592.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000000000-0000000000.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000000000-0000023296.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000000000-0000046592.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000023296-0000000000.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000023296-0000023296.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000023296-0000046592.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000046592-0000000000.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000046592-0000023296.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000046592-0000046592.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000069888-0000000000.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000069888-0000023296.tif", 
"http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/19Q/S2_percentile_30_UTM_438-0000069888-0000046592.tif"
]
        
if __name__ == "__main__":
    
    path = sys.argv[1]

    #make directory for files
    os.system("mkdir -p {}".format(path))

    #split tiles into managable sizes to fit in memeory 
    split_tiles(path, file_list, 6000)
