import osmnx as ox
import matplotlib.pyplot as plt
import geopandas
import json
import shapely
from shapely.geometry import Polygon, MultiPolygon

def create_geojson(geometry, filename):
    buildings = ox.geometries_from_polygon(geometry, tags={'building':True})

    list_polygons = []
    for i in buildings['geometry']:
        #polygon = geopandas.GeoSeries([i]).__geo_interface__
        #list_polygons+=str(polygon)
        list_polygons.append(geopandas.GeoSeries([i]).__geo_interface__)

    json_out = json.dumps(list_polygons)

    with open("{}.geojson".format(filename), "w") as text_file:
        text_file.write(json_out)

        
### example: ###  
        
geometry = Polygon([[-61.39543435084342,15.312346074078288],
[-61.39543435084342,15.292973781453892],
[-61.36711022364616,15.2930565728416],
[-61.366938562269205,15.312346074078288],
[-61.39543435084342,15.312346074078288]])

create_geojson(geometry, 'roseau')

