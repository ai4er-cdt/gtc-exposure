import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import gdal
import pyproj
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import transform
try:
    import rioxarray as rxr
except ModuleNotFoundError: 
    os.system("pip install rioxarray")
import rioxarray as rxr


def retrieve_image(image_dir, out_path, polygons):
    
    columns = ['Filename', 'Bands', 'Width', 'Height', 'Coordinates']
    dataframe = pd.DataFrame(columns = columns)
    
    for file in os.listdir(image_dir):
        if file.endswith('.tiff') or file.endswith('.tif') and file.startswith('tile'):
            #corrupted file- will look into this
            try:
                dataset = rasterio.open(image_dir+ file)
            except:
                pass
                
            left= dataset.bounds.left
            bottom = dataset.bounds.bottom
            right = dataset.bounds.right
            top = dataset.bounds.top

            coordinates = transform_bounds(dataset.crs, 'EPSG:4326', left, bottom, right, top)
            left, bottom, right, top = coordinates
            Geometry = (Polygon([ [left, top],
                                  [right, top],
                                  [right ,bottom],
                                  [left, bottom]]))

            dataframe.loc[len(dataframe)]= [file, dataset.count, dataset.width, dataset.height, Geometry]
    
    
    count = 0
    os.system("mkdir {}".format(out_path))
    
    for polygon in polygons:
        interceptions = []
        for i in range(len(dataframe)):
            if dataframe.loc[i]['Coordinates'].intersects(polygon):
                row = [i for i in dataframe.loc[i]]
                file = row[0]
                interceptions.append(file)
                print('Filename: {} \nBands: {} \nWidth: {} \nHeight: {} \nCoordinates: {}'.format(file,
                                                                                                   row[1],
                                                                                               row [2],
                                                                                               row[3],
                                                                                               row[4]))
     
        project = pyproj.Transformer.from_proj(
            pyproj.Proj(init='epsg:4326'), # source coordinate system
            pyproj.Proj(init='epsg:32617'))
        poly = transform(project.transform, polygon)


        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'},
        }

        os.system("mkdir {}".format('poly_shapes'))

        with fiona.open('poly_shapes/'+'poly.shp', 'w', 'ESRI Shapefile', schema) as c:
            ## If there are multiple geometries, put the "for" loop here
            c.write({
                'geometry': mapping(poly),
                'properties': {'id': 123},
            })

        aoi = os.path.join('poly_shapes', "poly.shp")
        poly = gpd.read_file(aoi)
        with fiona.open(aoi, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        for file in interceptions:
            src = rasterio.open(image_dir+ file)
           
            try:
                out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
                out_meta = src.meta
                out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})

                if 'train' in out_path:
                    name = 'train'
                else:
                    name ='test'
                with rasterio.open('{}{}_image_{}.tif'.format(out_path, name, count), "w", **out_meta) as dest:
                    dest.write(out_image)

                s2_cloud_free = rxr.open_rasterio('{}{}_image_{}.tif'.format(out_path, name, count), masked=True).squeeze()

                if np.isnan(s2_cloud_free).any()!=True:
                    os.system("rm {}{}_image_{}.tif".format(out_path, name, count))

                if count==0:
                    red = out_image[0]/out_image.max()
                    green = out_image[1]/out_image.max()
                    blue = out_image[2]/out_image.max()
                    s2_cloud_free_norm = np.dstack((red, green, blue))

                    plt.figure(figsize=(20,10))
                    plt.imshow(s2_cloud_free_norm)
                    plt.show()

                count+=1
            except:
                pass
            
        os.system("rm -r {}".format('poly_shapes'))

def split_tiles(path, tile_size):
    count=0
    for file in os.listdir(path):
        if file.endswith('.tif') and file.startswith('test') or file.startswith('train'):
            tile_size_x, tile_size_y = tile_size, tile_size

            ds = gdal.Open(path + file)
            try:
                band = ds.GetRasterBand(1)
                xsize = band.XSize
                ysize = band.YSize

                if 'train' in path:
                    name = 'train'
                else:
                    name ='test'
                output_filename = '{}tile_{}_'.format(name,count)

                for i in range(0, xsize, tile_size_x):
                    for j in range(0, ysize, tile_size_y):
                        com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(path) + str(file) + " " + str(path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
                        os.system(com_string)

                os.system("rm {}{}".format(path, file))
                count+=1
            except:
                pass

def polygons():
    # create list of polygons to produce training data
    polygons = []
    polygons.append(Polygon([[-66.0775927670462,18.440956583224942], 
    [-66.07879439668487,18.422065340910347], 
    [-66.04291716890167,18.420925285748748], 
    [-66.04411879854034,18.44291073117291], 
    [-66.07742110566925,18.444213484123757], 
    [-66.0775927670462,18.440956583224942]]))

    polygons.append(Polygon([[-66.12032364607379,18.469545047573636], 
    [-66.11274908781573,18.469585752437265], 
    [-66.11268471479937,18.466186863057448], 
    [-66.12047384977862,18.465983334610513], 
    [-66.12032364607379,18.469545047573636]]))

    polygons.append(Polygon([[-76.81544154191762,17.973422489287763], 
    [-76.79561465287954,17.973259204952388], 
    [-76.7953571608141,17.993586943788326], 
    [-76.81518404985219,17.9939134747012], 
    [-76.81544154191762,17.973422489287763]]))

    polygons.append(Polygon([[-77.92165047007306,18.48726791081117], 
    [-77.92096382456525,18.467079194198217], 
    [-77.87701851206525,18.467730480218645], 
    [-77.87633186655744,18.489872733322553], 
    [-77.92165047007306,18.48726791081117]]))

    polygons.append(Polygon([[-78.16361257076558,18.30775680441509], 
    [-78.16532918453511,18.260651384422882], 
    [-78.11571904659566,18.26032535079051], 
    [-78.11537572384175,18.307267886566336], 
    [-78.16361257076558,18.30775680441509]]))

    polygons.append(Polygon([[-78.30783422184228,18.370291058124224], 
    [-78.30817754459619,18.357094565229936], 
    [-78.308864190104,18.34748167002288], 
    [-78.286891533854,18.346829928949653], 
    [-78.2865482111001,18.370291058124224], 
    [-78.30783422184228,18.370291058124224]]))

    polygons.append(Polygon([[-78.35387142781204,18.278767976693885], 
    [-78.32271488789505,18.278523477050463], 
    [-78.32340153340286,18.257006159328974], 
    [-78.35515888813919,18.257739748161775], 
    [-78.35387142781204,18.278767976693885]]))

    polygons.append(Polygon([[-77.80665757539174,18.348880793248394], 
    [-77.80785920503041,18.331201588862175], 
    [-77.78138043763539,18.33095716322731], 
    [-77.7817237603893,18.34932886018125], 
    [-77.80665757539174,18.348880793248394]]))

    polygons.append(Polygon([[-76.87420390388037,17.951428493492195], 
    [-76.87351725837256,18.002045662028607], 
    [-76.92364238044287,18.00449451167833], 
    [-76.930508835521,17.934933961813915], 
    [-76.87677882453467,17.93379071947965], 
    [-76.87420390388037,17.951428493492195]]))

    polygons.append(Polygon([[-72.56209507419105,18.230462752087185], 
    [-72.49549045993324,18.22948447944948], 
    [-72.49480381442542,18.249374941424495], 
    [-72.55900516940589,18.250190075743266], 
    [-72.56209507419105,18.230462752087185]]))

    polygons.append(Polygon([[-73.40465777478241,18.272471579747418], 
    [-73.40500109753631,18.284941057176592], 
    [-73.38114016613983,18.286000513937058], 
    [-73.3812259968283,18.278910180020983], 
    [-73.39289897046112,18.278910180020983], 
    [-73.39272730908417,18.272716087919754], 
    [-73.40465777478241,18.272471579747418]]))

    polygons.append(Polygon([[-72.3185896196442,18.66191023753159], 
    [-72.23207228565983,18.662886050561045], 
    [-72.23241560841373,18.609858749665477], 
    [-72.31687300587467,18.61343779976815], 
    [-72.3185896196442,18.66191023753159]]))

    polygons.append(Polygon([[-73.19623713575962,19.907971950852737], 
    [-73.18876986586216,19.90809300307389], 
    [-73.18812613569858,19.894211744888686], 
    [-73.1955504902518,19.894252099147703], 
    [-73.19623713575962,19.907971950852737]]))

    polygons.append(Polygon([[-73.31975944552794,19.82939983669407], 
    [-73.27547081027403,19.83004576750048], 
    [-73.27641494784727,19.808486408574204], 
    [-73.32190521273985,19.808970920658027], 
    [-73.32216270480528,19.828996128606594], 
    [-73.31975944552794,19.82939983669407]]))

    polygons.append(Polygon([[-72.22984812512075,19.782717293839497], 
    [-72.20049402966177,19.78247499802398], 
    [-72.20049402966177,19.747095856134912], 
    [-72.19989321484243,19.739502049509234], 
    [-72.21637270702993,19.73925968803081], 
    [-72.21568606152212,19.774075181940386], 
    [-72.22890398754751,19.774236721040854], 
    [-72.22984812512075,19.782717293839497]]))

    polygons.append(Polygon([[-71.72472614717672,19.563940116204773], 
    [-71.72498363924215,19.541697807950403], 
    [-71.68970722627829,19.5421022409221], 
    [-71.69022221040915,19.563859240640078], 
    [-71.72472614717672,19.563940116204773]]))

    polygons.append(Polygon([[-70.72640207002964,19.818864659703387], 
    [-70.72708871553745,19.781555232544534], 
    [-70.69327142427768,19.77153995863976], 
    [-70.65447595308628,19.77299382777599], 
    [-70.69344308565464,19.801422379918886], 
    [-70.72640207002964,19.818864659703387]]))

    polygons.append(Polygon([[-70.74723775465475,19.7552811845663], 
    [-70.74655110914694,19.716260105324974], 
    [-70.69582517225729,19.714644120142577], 
    [-70.69754178602682,19.75269623205292], 
    [-70.74723775465475,19.7552811845663]]))

    polygons.append(Polygon([[-70.59331894887129,19.7553273376829], 
    [-70.57692528737226,19.756539019405476], 
    [-70.57671071065107,19.74139233653531], 
    [-70.58769703877607,19.741755873759207], 
    [-70.58709622395673,19.734242602926823], 
    [-70.60344697011152,19.734888919689876], 
    [-70.60190201771894,19.748541749980696], 
    [-70.59610844624677,19.748824488562562], 
    [-70.59653759968916,19.7553273376829], 
    [-70.59331894887129,19.7553273376829]]))

    polygons.append(Polygon([[-70.69011205425936,19.502944993010054], 
    [-70.69285863629061,19.413276233740607], 
    [-70.6317471860953,19.41392383918573], 
    [-70.60565465679842,19.41360003678563], 
    [-70.60908788433748,19.502297742951956], 
    [-70.69011205425936,19.502944993010054]]))

    polygons.append(Polygon([[-71.11254871584644,18.251276292049287], 
    [-71.07958973147144,18.187358569703854], 
    [-71.143447763698,18.183444477720833], 
    [-71.143447763698,18.24997208345702], 
    [-71.11254871584644,18.251276292049287]]))

    polygons.append(Polygon([[-71.42888353789733,18.50406277098119], 
    [-71.42905519927429,18.47557273253204], 
    [-71.40794084990905,18.475247104719386], 
    [-71.40828417266296,18.504225557585517], 
    [-71.42888353789733,18.50406277098119]]))

    polygons.append(Polygon([[-76.98896827893832,18.055265190442288],
    [-76.99034156995394,18.019192428803905],
    [-77.01128425794222,18.019518912287356],
    [-77.01145591931918,18.002377711518832],
    [-76.94279136853793,17.999602312692026],
    [-76.94227638440707,18.033067443052552],
    [-76.96871223645785,18.03469972577332],
    [-76.96802559095003,18.05493877328137],
    [-76.98896827893832,18.055265190442288]]))

    polygons.append(Polygon([[-77.24542906719492,17.903356881053103], 
    [-77.23959258037851,17.84797296203047], 
    [-77.21641829448984,17.84862656022137], 
    [-77.2227697654371,17.90352022989205], 
    [-77.24542906719492,17.903356881053103]]))

    polygons.append(Polygon([[-77.61725941872518,18.00500499383862], 
    [-77.54928151345175,18.005658012776067], 
    [-77.53108540749471,17.931851583193883], 
    [-77.5688509104244,17.933484802157444], 
    [-77.56713429665487,17.869965684546454], 
    [-77.64953175759237,17.866698061695857], 
    [-77.65021840310018,17.91929948087438], 
    [-77.60283986306112,17.9196261490219], 
    [-77.61725941872518,18.00500499383862]]))


    polygons.append(Polygon([[-77.65122856868395,17.97250230579589],
    [-77.64985527766832,17.8660083835417],
    [-77.74117913020739,17.862740687940736],
    [-77.74049248469957,17.873850607878044],
    [-77.78306450618395,17.875157611618686],
    [-77.78512444270739,17.97511484824287],
    [-77.65122856868395,17.97250230579589]]))

    polygons.append(Polygon([[-77.89764498925625,18.036904437768282], 
    [-77.84442996240078,18.02188705388032], 
    [-77.8454599306625,18.071831219215643], 
    [-77.91240786767422,18.092718685321067], 
    [-77.91515444970547,18.06758814893465], 
    [-77.89764498925625,18.036904437768282]]))

    polygons.append(Polygon([[-77.58973327967041,20.053238388810396],
    [-77.59213653894776,20.029048379911583],
    [-77.56209579798096,20.028564541740096],
    [-77.56295410486572,20.055657184756555],
    [-77.58973327967041,20.053238388810396]]))

    polygons.append(Polygon([[-77.65850388961779,19.968218733673265], 
    [-77.63017976242052,19.96870275734344], 
    [-77.62940728622424,19.93175134763844], 
    [-77.66391122299181,19.931832036992787], 
    [-77.6606496568297,19.93586645216965], 
    [-77.65850388961779,19.968218733673265]]))

    polygons.append(Polygon([[-77.59690543176107,20.014360189870708],
    [-77.51776953698568,20.015166658848656],
    [-77.51828452111654,19.96047870558841],
    [-77.60171195031576,19.96257623369726],
    [-77.59690543176107,20.014360189870708]]))

    polygons.append(Polygon([[-77.3080648579231,19.911148611579883], 
    [-77.30797902723462,19.89944669462917], 
    [-77.36574308057935,19.89960810626574], 
    [-77.36522809644849,19.928820900195152], 
    [-77.30686322828443,19.92865951836999], 
    [-77.3080648579231,19.911148611579883]]))

    polygons.append(Polygon([[-77.3632512819183,20.200719056302848],
    [-77.26368768328548,20.20474656073141],
    [-77.26609094256283,20.24678748203499],
    [-77.36239297503353,20.243083183273413],
    [-77.3632512819183,20.200719056302848]]))

    polygons.append(Polygon([[-77.15627585361977,20.311015929740663],
    [-77.05104742954751,20.309245053233315],
    [-77.04967413853188,20.35318899750564],
    [-77.1524993033268,20.352545220364455],
    [-77.15627585361977,20.311015929740663]]))

    polygons.append(Polygon([[-76.60715148562802,21.199068951113308],
    [-76.46055266971005,21.19458762325338],
    [-76.45952270144834,21.067773857083314],
    [-76.66414306277646,21.070977559002912],
    [-76.6665463220538,21.196828304174414],
    [-76.60715148562802,21.199068951113308]]))

    polygons.append(Polygon([[-76.6779283661015,20.411245221211],
    [-76.67998830262493,20.334001857994995],
    [-76.57321492616009,20.33110448002253],
    [-76.573558248914,20.412532282791197],
    [-76.6779283661015,20.411245221211]]))

    polygons.append(Polygon([[-76.30902653859421,20.914846718989207],
    [-76.30954152272507,20.86464847539306],
    [-76.28859883473679,20.845238268538623],
    [-76.2094629399614,20.84491741763076],
    [-76.21186619923874,20.913724258118222],
    [-76.30902653859421,20.914846718989207]]))

    polygons.append(Polygon([[-76.15700464107113,20.61642923627211],
    [-76.15906457759456,20.57963183512259],
    [-76.10936860896663,20.579872895026455],
    [-76.11091356135921,20.61377819213201],
    [-76.15700464107113,20.61642923627211]]))

    polygons.append(Polygon([[-75.85695325397705,20.059333313052147],
    [-75.86072980427002,19.97933392339478],
    [-75.75567304157471,19.977075285873703],
    [-75.75670300983643,20.074812361476074],
    [-75.85695325397705,20.073844965707842],
    [-75.85695325397705,20.059333313052147]]))

    polygons.append(Polygon([[-75.2254256497316,20.169879582796238],
    [-75.23074715241715,20.123304868483338],
    [-75.18045036896989,20.121854221920337],
    [-75.17753212556168,20.17116865061217],
    [-75.2254256497316,20.169879582796238]]))

    polygons.append(Polygon([[-74.9596754299022,20.674541490778875],
    [-74.96053373678697,20.639926834990018],
    [-74.92869055136217,20.640167799511307],
    [-74.92877638205064,20.660568077597933],
    [-74.94611418112291,20.674541490778875],
    [-74.9596754299022,20.674541490778875]]))

    polygons.append(Polygon([[-74.5200985913285,20.357256297291237],
    [-74.51984109926308,20.331986696049643],
    [-74.4856804852494,20.33029654346995],
    [-74.48576631593788,20.340598138959216],
    [-74.49872674989784,20.35427888393177],
    [-74.50207414674843,20.354198412503735],
    [-74.50207414674843,20.349933366820327],
    [-74.5061940197953,20.350013840470435],
    [-74.50610818910683,20.35749770666927],
    [-74.5200985913285,20.357256297291237]]))

    polygons.append(Polygon([[-74.1804506146205,20.259804792247824],
    [-74.17890566222792,20.227915391407688],
    [-74.13736360900526,20.229526124125037],
    [-74.14989488952284,20.26109311527556],
    [-74.1804506146205,20.259804792247824]]))
    
    return polygons

def test_areas():
    test = []
    
    test.append(Polygon([[-77.13783521566255,18.418600041643828], 
                         [-77.07517881307466,18.418437173802708], 
                         [-77.0746638289438,18.380321862478468], 
                         [-77.13869352254731,18.379996054245137], 
                         [-77.13783521566255,18.418600041643828]]))

    test.append(Polygon([[-72.34347052937672,18.58619943673911], 
                          [-72.30810828572437,18.58644349918482], 
                          [-72.30827994710133,18.55528203493343], 
                          [-72.34321303731129,18.556909406316695], 
                          [-72.34347052937672,18.58619943673911]]))


    test.append(Polygon([[-70.00152048333261,18.46316824984316], 
                         [-69.97555670006845,18.4635753131738], 
                         [-69.97594168277088,18.437337401095554], 
                         [-70.00169088931385,18.437174551000332], 
                         [-70.00152048333261,18.46316824984316]]))

    test.append(Polygon([[-66.0664389311957,18.447325116984587], 
                          [-66.06575228568789,18.423385888495915], 
                          [-66.04026057121035,18.423630183172026], 
                          [-66.0415480315375,18.4458595457671], 
                          [-66.06618143913028,18.4449639127564], 
                          [-66.0664389311957,18.447325116984587]]))

    test.append(Polygon([[-76.81149151220227,17.970649199371998], 
                        [-76.77999164953137,17.96975111955906], 
                        [-76.780561259368,17.997712511314443], 
                        [-76.81231861410433,17.997385987434477], 
                        [-76.81149151220227,17.970649199371998]]))
    
    return test


def informal_settlements():
    informal = []

    informal.append(Polygon([[-77.1178623628689,18.397458368293538], 
                            [-77.11305584431422,18.397438007321558], 
                            [-77.10782017231715,18.39782486537758], 
                            [-77.10408653736842,18.399087238254033], 
                            [-77.10333551884425,18.394953950684403], 
                            [-77.10520233631861,18.389028720179283], 
                            [-77.11726154804957,18.388702928400765], 
                            [-77.11951460362208,18.393569378980867], 
                            [-77.1178623628689,18.397458368293538]]))

    informal.append(Polygon([[-72.33905729667333,18.587105051612888], 
                            [-72.33910021201757,18.57514562714193], 
                            [-72.32133325950292,18.575023587873503], 
                            [-72.32103285209325,18.587227082230804], 
                            [-72.32089101418691,18.59293111081294], 
                            [-72.33840047463613,18.59293111081294], 
                            [-72.33905729667333,18.587105051612888]]))

    informal.append(Polygon([[-69.98864896560471,18.45767578823935], 
                            [-69.98849876189988,18.452892568733045], 
                            [-69.98341329360764,18.452994340748322], 
                            [-69.98328454757493,18.454073120401635], 
                            [-69.9822545793132,18.454154537458894], 
                            [-69.98229749465744,18.458489939989008], 
                            [-69.98856313491623,18.45834746371158], 
                            [-69.98864896560471,18.45767578823935]]))

    informal.append(Polygon([[-66.057148704191,18.433459220520398], 
                             [-66.05684829678133,18.428492111145008], 
                             [-66.04993892635896,18.4284513965408], 
                             [-66.05019641842439,18.43292994520503], 
                             [-66.05701995815828,18.432767090934, 
                             ]]))


    informal.append(Polygon([[-76.80992497084307,17.989681156780875], 
                            [-76.80966747877764,17.982007336502964], 
                            [-76.80061234114336,17.98217061274875], 
                            [-76.80078400252032,17.98988524319038],
                            [-76.80992497084307,17.989681156780875]]))
    
    
    ≈.append(Polygon([[-76.80361641524004,17.978047841309895], 
                            [-76.80348766920733,17.97159816419407], 
                            [-76.79674996016192,17.971802271509187], 
                            [-76.79700745222735,17.978211121217566], 
                            [-76.80361641524004,17.978047841309895]]))

    return informal


if __name__ == "__main__":
    
    
    def split_tiles(path, file_list, tile_size=8000):
       
        count=0
        for url in file_list:
            os.system("cd {} && wget {}".format(path, url))

            file = url.split('/')[-1]

            output_filename = 'tile_{}_'.format(count)

            tile_size_x, tile_size_y = tile_size, tile_size

            ds = gdal.Open(path + file)
            try:
                band = ds.GetRasterBand(1)
            except:
                pass
            xsize = band.XSize
            ysize = band.YSize

            for i in range(0, xsize, tile_size_x):
                for j in range(0, ysize, tile_size_y):
                    com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(path) + str(file) + " " + str(path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
                    os.system(com_string)

            os.system("rm {}{}".format(path, file))
            count+=1 
            print("\nCOMPLETED {} OF {} FILES\n".format(count, len(file_list)))

    
    path = sys.argv[1]
    
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

    #make directory for files
    os.system("mkdir -p {}".format(path))

    #split tiles into managable sizes to fit in memeory 
    split_tiles(path, file_list)