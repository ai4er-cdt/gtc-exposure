from ipyleaflet import DrawControl
from descarteslabs import workflows as wf
import descarteslabs as dl


class FullArray(object):
    def __init__(self, map_widget, variable=None, draw_control=None):
        self.m = map_widget
        if draw_control is None:
            self.draw_control = DrawControl()
            self.draw_control.circle = {}
            self.draw_control.polygon = {}
            self.draw_control.circlemarker = {}
            self.draw_control.polyline = {}
            self.draw_control.rectangle = {
                "shapeOptions": {
                    "fillColor": "#000000",
                    "color": "#fca45d",
                    "fillOpacity": 0.0,
                }
            }
            self.m.add_control(self.draw_control)
        else:
            self.draw_control = draw_control
        self.variable = variable
        self.draw_control.on_draw(self.calculate)

    def calculate(self, *args, **kwargs):
        last_draw = self.draw_control.last_draw

        if last_draw["geometry"]["type"] == "Point":
            last_draw_context = wf.GeoContext.from_dltile_key(
                dl.raster.dltile_from_latlon(
                    self.draw_control.last_draw["geometry"]["coordinates"][1],
                    self.draw_control.last_draw["geometry"]["coordinates"][0],
                    156543.00 / 2 ** (max(self.m.zoom, 0)),
                    2,
                    0,
                ).properties.key
            )

        elif last_draw["geometry"]["type"] == "Polygon":
            last_draw_context = wf.GeoContext(
                geometry=last_draw["geometry"],
                resolution=156543.00 / 2 ** (max(self.m.zoom, 0)),
                crs="EPSG:3857",
                bounds_crs="EPSG:4326",
            )

        data = self.variable.compute(last_draw_context)
        self._data = data.ndarray

    _data = None

    @property
    def data(self):
        if self._data is not None:
            return self._data
        else:
            raise RuntimeError(
                "To display the deforested acreage please draw a polygon first"
            )
