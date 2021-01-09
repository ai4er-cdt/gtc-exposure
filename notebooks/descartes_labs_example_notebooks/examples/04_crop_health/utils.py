# flake8: noqa
# from IPython.display import display, IFrame, IFrame
from ipyleaflet import DrawControl
import ipywidgets as widgets
from collections import defaultdict
from descarteslabs import workflows as wf
import descarteslabs as dl
import arrow
import pandas as pd
import numpy as np

DEBUG_VIEW = widgets.Output(layout={"border": "1px solid black"})


class Auger(object):
    def __init__(self, map_widget, variable=None, draw_control=None):
        self.m = map_widget
        if draw_control is None:
            draw_control = DrawControl(
                edit=False,
                remove=False,
                circlemarker={},
                polyline={},
                polygon={},
                rectangle={
                    "shapeOptions": {
                        "fillColor": "#d534eb",
                        "color": "#d534eb",
                        "fillOpacity": 0.2,
                    }
                },
            )
            self.draw_control = draw_control
            self.m.add_control(self.draw_control)
        else:
            self.draw_control = draw_control
        self.variable = variable
        self.storage = defaultdict(dict)
        self.draw_control.on_draw(self.calculate)

    def calculate(self, *args, **kwargs):
        last_draw = self.draw_control.last_draw

        if last_draw["geometry"]["type"] == "Point":
            auger_context = wf.GeoContext.from_dltile_key(
                dl.raster.dltile_from_latlon(
                    self.draw_control.last_draw["geometry"]["coordinates"][1],
                    self.draw_control.last_draw["geometry"]["coordinates"][0],
                    156543.00 / 2 ** (max(self.m.zoom, 0)),
                    2,
                    0,
                ).properties.key
            )

        elif last_draw["geometry"]["type"] == "Polygon":
            auger_context = wf.GeoContext(
                geometry=last_draw["geometry"],
                resolution=156543.00 / 2 ** (max(self.m.zoom, 0)),
                crs="EPSG:3857",
                bounds_crs="EPSG:4326",
            )

        with self.m.output_log:
            timeseries = self.variable.map(
                lambda img: (img.properties["date"], img.median(axis="pixels"))
            ).compute(auger_context)
            self.timeseries = timeseries

        values = defaultdict(list)
        dates = []

        with self.m.output_log:
            for date, valdict in timeseries:
                # if any(vv == np.ma.masked for vv in valdict.values()):
                #    print("Skipping {}".format(date))
                #    continue
                for k, v in valdict.items():
                    values[k].append(v)
                dates.append(arrow.get(date).datetime)

        self.storage["dates"] = dates
        for k, v in values.items():
            self.storage[k] = v

        self._df = pd.DataFrame.from_dict(self.storage)

    _df = None

    @property
    def df(self):
        if self._df is not None:
            return self._df
        else:
            raise RuntimeError("Must click on a point first")
