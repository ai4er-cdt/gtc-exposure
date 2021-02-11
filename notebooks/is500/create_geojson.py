import osmnx as ox
import matplotlib.pyplot as plt
import geopandas
import json
import shapely
from shapely.geometry import Polygon, MultiPolygon
​
def create_geojson(geometry, filename):
    buildings = ox.geometries_from_polygon(geometry, tags={'building':True})
​
    list_polygons = []
    for i in buildings['geometry']:
        #polygon = geopandas.GeoSeries([i]).__geo_interface__
        #list_polygons+=str(polygon)
        list_polygons.append(geopandas.GeoSeries([i]).__geo_interface__)
​
    json_out = json.dumps(list_polygons)
​
    with open("{}.geojson".format(filename), "w") as text_file:
        text_file.write(json_out)
​
        
### example: ###  
        
#geometry = Polygon([[-77.11751845981902,18.39737744609368], 
#         [-77.10322765018768,18.397866108928678],
#         [-77.10245517399139,18.387929692189733],
#         [-77.11769012119598,18.387563174058496],
#         [-77.11751845981902,18.39737744609368]])
        
#create_geojson(geometry, 'parry_town')
