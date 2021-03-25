from ipyleaflet import Map, basemaps, LocalTileLayer, ImageOverlay, ScaleControl, LayersControl
import os

os.system('jupyter labextension install @jupyter-widgets/jupyterlab-manager')
os.system('!jupyter labextension install jupyter-leaflet')

def exposure_quantification():
    base = os.getcwd()+'/exposure_quantification/'

    center = [18.51097696266426, -72.29284267872913]
    zoom = 16

    rel_url_S2 = base + 'GHS_S2_Haiti.png'

    rel_url_BU = base + 'GHS_BU_2.png'

    rel_url_POP = base + 'GHS_POP_2.png'
    
    bounds = ((18.504317897,-72.309451052),(18.523663424,-72.274449289))

    return center, zoom, rel_url_S2, rel_url_BU, rel_url_POP, bounds


