import IPython
import ipyleaflet

import matplotlib.dates as mdates
import ipywidgets as widgets

from collections import defaultdict

import descarteslabs as dl
import descarteslabs.workflows as wf

import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd

import datetime
import arrow


class FieldNDVI(object):
    def __init__(self, map_widget):
        self.m = map_widget

        # Rectangle & Polygon draw control
        self.draw_control = ipyleaflet.DrawControl(
                            edit=False,
                            remove=False,
                            circlemarker={},
                            polyline={},
                            polygon={},
                            rectangle={"shapeOptions": {
                                        "fillColor": "#d534eb",
                                        "color": "#d534eb",
                                        "fillOpacity": 0.2
                                        }}
                            )
        self.m.add_control(self.draw_control)

        # Datepicker control
        sd = widgets.DatePicker(
                    description='Start date',
                    disabled=False
                )
        ed = widgets.DatePicker(
                    description='End date',
                    disabled=False
                )
        date_box = widgets.VBox([sd, ed])
        self.dt_control = ipyleaflet.WidgetControl(widget=date_box, position='bottomleft')

        self.m.add_control(self.dt_control)

        # Setting default values for dates
        self.dt_control.widget.children[0].value = datetime.date(2018, 3, 1)
        self.dt_control.widget.children[1].value = datetime.date(2019, 12, 1)
        # self.dt_control.widget.children[0].value = datetime.date(2018, 10, 1)
        # self.dt_control.widget.children[1].value = datetime.date(2019, 10, 1)

        # Adding clear plot button
        clear_plot_button = widgets.Button(description='Clear plot',
                                           disabled=False,
                                           button_style='warning',
                                           tooltip='Plot and all polygons will be cleared')
        self.clear_plot_control = ipyleaflet.WidgetControl(widget=clear_plot_button,
                                                           position='topright')
        self.m.add_control(self.clear_plot_control)

        self.get_s2_collection()

        self.ax = None
        self.fig = None
        self.storage = defaultdict(dict)
        self.draw_control.on_draw(self.get_s2_collection)
        self.draw_control.on_draw(self.calculate)

        self.fig_output = widgets.Output()
        self.fig_widget = ipyleaflet.WidgetControl(widget=self.fig_output, position='bottomright')
        self.m.add_control(self.fig_widget)

        self.draw_control.on_draw(self.plot_timeseries)
        self.clear_plot_control.widget.on_click(self.clear_plot)

    def get_s2_collection(self, *args, **kwargs):
        s2_new = wf.ImageCollection.from_id('sentinel-2:L1C',
                                            start_datetime=self.dt_control.widget.children[0].value,
                                            end_datetime=self.dt_control.widget.children[1].value)
        s2_new = s2_new.filter(lambda img: img.properties['cloud_fraction_0'] < 0.05)
        red, nir = s2_new.unpack_bands("red nir")
        ndvi = (nir - red)/(nir + red)
        self.variable = ndvi

    def plot_timeseries(self, *args, **kwargs):

        if self.ax is None or self.fig is None:
            fig, ax = plt.subplots(figsize=[5, 4])
            ax.cla()
            ax.set_visible(True)
            self.fig = fig
            self.ax = ax
            first_draw = True
        else:
            first_draw = False

        df = self.df.drop_duplicates(subset=['dates'])
        dates = df['dates'].values.astype(np.datetime64)
        ndvi = df['nir_sub_red_div_nir_add_red'].values.astype(float)

        # Setting up interpolation
        target_dates = np.arange(dates.min(),
                                 dates.max(),
                                 datetime.timedelta(days=1))

        base_dt = np.datetime64('1970-01-01T00:00:00Z')
        x1 = (dates - base_dt) / np.timedelta64(1, 's')
        x2 = (target_dates - base_dt) / np.timedelta64(1, 's')

        data = np.array([(x, y) for x, y in sorted(zip(x1, ndvi))])
        self.dates = dates
        self.data = data

        # Interpolation
        y2 = interpolate.pchip_interpolate(data[:, 0], data[:, 1], x2)

        _ = self.ax.scatter(dates, ndvi)
        _ = self.ax.plot(target_dates, y2)

        # format the ticks
        # self.ax.xaxis.set_major_locator(mdates.MonthLocator())
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        _ = self.ax.set_xlabel("Date")
        _ = self.ax.set_ylabel("NDVI")
        _ = self.ax.set_ylim(0, 1.0)
        self.fig.autofmt_xdate()

        if not first_draw:
            with self.fig_output:
                IPython.display.clear_output()

        with self.fig_output:
            IPython.display.display(self.fig)

        return ""

    def calculate(self, *args, **kwargs):
        last_draw = self.draw_control.last_draw

        if last_draw['geometry']['type'] == 'Point':
            auger_context = wf.GeoContext.from_dltile_key(
                        dl.raster.dltile_from_latlon(
                            self.draw_control.last_draw['geometry']['coordinates'][1],
                            self.draw_control.last_draw['geometry']['coordinates'][0],
                            156543.00/2**(max(self.m.zoom, 0)), 2, 0).properties.key)

        elif last_draw['geometry']['type'] == 'Polygon':
            auger_context = wf.GeoContext(
                geometry=last_draw['geometry'],
                resolution=156543.00/2**(max(self.m.zoom, 0)),
                crs='EPSG:3857',
                bounds_crs='EPSG:4326',
            )

        with self.m.output_log:
            timeseries = (
                    self.variable.map(lambda img: (img.properties['date'], img.median(axis='pixels')))
                    .compute(auger_context)
            )
            self.timeseries = timeseries

        values = defaultdict(list)
        dates = []

        with self.m.output_log:
            for date, valdict in timeseries:
                for k, v in valdict.items():
                    values[k].append(v)
                dates.append(arrow.get(date).datetime)

        self.storage['dates'] = dates
        for k, v in values.items():
            self.storage[k] = v

        self._df = pd.DataFrame.from_dict(self.storage)

    def clear_plot(self, *args, **kwargs):
        # Clear draw control polygons
        self.draw_control.clear()

        # Clear plot
        with self.fig_output:
            IPython.display.clear_output()

        # Clear axes and fig
        self.ax = None
        self.fig = None

    _df = None

    @property
    def df(self):
        if self._df is not None:
            return self._df
        else:
            raise RuntimeError("Must click on a point first")
