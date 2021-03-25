from ipyleaflet import Map, basemaps, LocalTileLayer, ImageOverlay, ScaleControl, LayersControl
import os

os.system('jupyter labextension install @jupyter-widgets/jupyterlab-manager')
os.system('!jupyter labextension install jupyter-leaflet')

def exposure_quantification():
    base = os.getcwd()+'/exposure_quantification/'

    #center = [18.5139906605, -72.2919501705]
    center = [18.51097696266426, -72.29284267872913]
    zoom = 16

    m = Map(basemap=basemaps.OpenStreetMap.Mapnik, center=center, zoom=zoom)

    m.add_control(ScaleControl(position='bottomleft'))

    rel_url_S2 = 'GHS_S2_Haiti.png'
    S2 = ImageOverlay(name='Sentinel-2 Composite', url=rel_url_S2, 
                      bounds=((18.504317897,-72.309451052),(18.523663424,-72.274449289)))
    m.add_layer(S2)

    rel_url_BU = 'GHS_BU_2.png'
    BU = ImageOverlay(name='GHSL Built-up Probability', url=rel_url_BU, 
                      bounds=((18.504317897,-72.309451052),(18.523663424,-72.274449289)), opacity=0.7)
    m.add_layer(BU)

    rel_url_POP = base+ 'GHS_POP_2.png'
    POP = ImageOverlay(name='GHSL Population Density', url=rel_url_POP, 
                      bounds=((18.504317897,-72.309451052),(18.523663424,-72.274449289)), opacity=0.5)
    m.add_layer(POP)

    #rel_url = 'GHS_S2_Haiti_1.png'
    #io1 = ImageOverlay(url=rel_url, bounds=((18.458481455,-72.389775449),(18.569945471,-72.188104178)))
    #m.add_layer(io1)

    control = LayersControl(position='topright')
    m.add_control(control)
    
    return m