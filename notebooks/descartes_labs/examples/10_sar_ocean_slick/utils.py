import ipywidgets as widgets
import descarteslabs.workflows as wf
import ipyleaflet
from datetime import datetime
import descarteslabs as dl
import time


class Slickfinder_Map(object):
    def __init__(
        self, map_widget, area="SB", false_color=False, scales=None, colormap=None
    ):

        self.false_color = false_color
        self.m = map_widget
        self.create_locations()

        self.product = "sentinel-1:GRD"
        if not false_color and not colormap:
            colormap = "gray"
        self.visualize_kwargs = {"scales": scales, "colormap": colormap}

        # Establishing a kernel for future filtering
        self.ker = wf.Kernel(
            dims=(5, 5),
            data=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                3.0,
                2.0,
                1.0,
                1.0,
                3.0,
                4.0,
                3.0,
                1.0,
                1.0,
                2.0,
                3.0,
                2.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        )

        self.cur_key = None

        #############################
        # Control widgets
        #############################
        self.area_selector = widgets.Dropdown(
            options=[
                ("Gulf of Mexico - Seep", "GoM"),
                ("Santa Barbara - Seep", "SB"),
                ("Caspian Sea - Facility Spill", "Caspian"),
                ("Persian Gulf - Facility Spill", "Kuwait"),
                ("Mediterranean Sea - Shipwreck", "Med"),
                ("Indonesia - Unknown", "Indonesia"),
            ],
            value=area,
            description="Area of Interest",
            disabled=False,
        )

        # Creating and adding the datetime controls
        self.sd = widgets.DatePicker(description="Start date")
        self.ed = widgets.DatePicker(description="End date")
        self.group_days = widgets.BoundedIntText(
            value=5, min=1, max=365, step=1, description="Group days", disabled=False
        )

        # The range is small, this gets around low sigfig visualization with a FloatSlider
        values = [i * 0.001 for i in range(201)]
        self.SAR_clip = widgets.SelectionSlider(
            description="Spill Clip Value",
            options=[(f"{i:.3f}", i) for i in values],
            value=values[5],
            style={"description_width": "initial", "widget_width": "initial"},
        )
        self.refresh_area(None, initial=True)
        self.recompute_layers = widgets.Button(description="Recompute Layers")

        date_box = widgets.VBox(
            [
                self.area_selector,
                self.sd,
                self.ed,
                self.group_days,
                self.SAR_clip,
                self.recompute_layers,
            ],
            layout=widgets.Layout(border="3px solid", overflow="hidden"),
        )
        self.dt_control = ipyleaflet.WidgetControl(widget=date_box, position="topright")
        self.m.add_control(self.dt_control)

        #############################
        # Carousel control
        #############################

        # Widgets for identifying scenes on the carousel
        txt_layout = widgets.Layout(max_width="175px")
        self.scene_selected = widgets.Text(
            description="Current", value="", layout=txt_layout
        )
        self.scene_prev = widgets.Text(
            description="Previous", value="", layout=txt_layout
        )
        self.scene_next = widgets.Text(description="Next", value="", layout=txt_layout)
        self.text_out = widgets.HBox(
            [self.scene_prev, self.scene_selected, self.scene_next],
            layout=widgets.Layout(border="3px solid", justify_content="space-between"),
        )
        # Add next/prev buttons
        self.next_scene_button = widgets.Button(description="Next Scene")
        self.prev_scene_button = widgets.Button(description="Previous Scene")
        self.buttons = widgets.HBox(
            [self.prev_scene_button, self.next_scene_button],
            layout=widgets.Layout(border="3px solid", justify_content="space-between"),
        )
        self.carousel_widget = widgets.VBox(
            [self.text_out, self.buttons], layout=widgets.Layout(max_width="600px")
        )
        self.carousel_widget_ctrl = ipyleaflet.WidgetControl(
            widget=self.carousel_widget, position="bottomright"
        )
        self.m.add_control(self.carousel_widget_ctrl)

        # Initializing carousel
        self.update_layers(None)

        # Listen for changes
        # self.sd.observe(self.update_layers, names='value')
        # self.ed.observe(self.update_layers, names='value')
        self.group_days.observe(self.update_layers, names="value")
        self.SAR_clip.observe(self.SAR_clip_update, names="value")
        self.area_selector.observe(self.refresh_area, names="value")
        self.next_scene_button.on_click(self.next_scene)
        self.prev_scene_button.on_click(self.prev_scene)
        self.recompute_layers.on_click(self.update_layers)

    def create_ic(self):
        # Getting bounds for geocontext for keys
        self.ctx = wf.GeoContext.from_dltile_key(
            dl.raster.dltile_from_latlon(
                self.locations[self.area_selector.value]["Lat"],
                self.locations[self.area_selector.value]["Lon"],
                156543.00 / 2 ** (max(wf.map.map.zoom, 0)),
                2,
                0,
            ).properties.key
        )

        self.ic = wf.ImageCollection.from_id(
            self.product, start_datetime=self.sd.value, end_datetime=self.ed.value
        )

        # Visualization by mapping vh vh vv as RGB
        self.vv, self.vh = self.ic.pick_bands("vv"), self.ic.pick_bands("vh")
        if self.false_color:
            self.vis_bands = self.vv.concat_bands(self.vv.concat_bands(self.vv))
        else:
            self.vis_bands = self.vv

        # Grouping by date (parameter) and getting keys
        self.icg = self.vv.groupby(
            lambda img: img.properties["date"]
            // wf.Timedelta(days=self.group_days.value)
        )
        self.keys = self.icg.groups.keys().compute(
            self.ctx,
            start_date=wf.Datetime.from_string(str(self.sd.value)),
            end_date=wf.Datetime.from_string(str(self.ed.value)),
            product=str(self.product),
            progress_bar=False,
        )

    def update_layers(self, change):

        self.create_ic()

        # Setting up minimap carousel

        self.view_layers = []

        for i in range(len(self.keys)):
            # Visualizing a day
            lyr = self.icg[self.keys[i]]
            lyr_min, lyr_max, lyr_mean = (
                lyr.min(axis="images"),
                lyr.max(axis="images"),
                lyr.mean(axis="images"),
            )
            if self.false_color:
                vis_lyr = lyr_min.concat_bands(lyr_mean.concat_bands(lyr_max))
            else:
                vis_lyr = lyr_min
            self.view_layers.append(vis_lyr)

        # Initialize with the latest image on main map
        self.cur_lyr_num = self.locations[self.area_selector.value]["lyr_num"]
        if self.cur_lyr_num == -1:
            self.cur_lyr_num = len(self.view_layers) - 1
        self.cur_lyr = self.view_layers[self.cur_lyr_num]

        self.plot_cur_scene()

    def plot_cur_scene(self):
        self.m.clear_layers()
        self.cur_lyr.visualize(
            name=str(self.keys[self.cur_lyr_num]),
            start_date=str(self.sd.value),
            end_date=str(self.ed.value),
            product=self.product,
            **self.visualize_kwargs,
        )

        # Change the selected scene output
        self.scene_selected.value = self.keys[self.cur_lyr_num][:10]
        self.scene_prev.value = (
            self.keys[self.cur_lyr_num - 1][:10] if self.cur_lyr_num - 1 >= 0 else ""
        )
        self.scene_next.value = (
            self.keys[self.cur_lyr_num + 1][:10]
            if self.cur_lyr_num + 1 < len(self.view_layers)
            else ""
        )
        self.cur_key = self.keys[self.cur_lyr_num]

        self.SAR_clip_update(None)

    def next_scene(self, target):
        if self.cur_lyr_num + 1 < len(self.view_layers):
            self.cur_lyr_num += 1
            self.cur_lyr = self.view_layers[self.cur_lyr_num]
            self.plot_cur_scene()

    def prev_scene(self, target):
        if self.cur_lyr_num - 1 >= 0:
            self.cur_lyr_num -= 1
            self.cur_lyr = self.view_layers[self.cur_lyr_num]
            self.plot_cur_scene()

    def SAR_clip_update(self, change):
        if change is None:
            clipval = self.SAR_clip.value
        else:
            clipval = change["new"]

        lyr = self.icg[self.cur_key].pick_bands("vv").min(axis="images")

        self.spill_mask = lyr.mask(lyr > clipval)
        self.spill_mask = wf.conv2d(self.spill_mask, self.ker)
        self.spill_mask.visualize(
            name="Spill Mask",
            scales=[(0.1, 0.3)],
            colormap="magma",
            map=self.m,
            checkerboard=False,
        )

    def create_locations(self):
        self.locations = {
            "GoM": {
                "Lat": 28.935833,
                "Lon": -88.97,
                "SD": "2018-01-01",
                "ED": "2018-06-27",
                "zoom": 11,
                "clip_val": 0.005,
                "group": 8,
                "lyr_num": -1,
            },
            "SB": {
                "Lat": 34.378366,
                "Lon": -119.813161,
                "SD": "2017-01-01",
                "ED": "2017-06-10",
                "zoom": 12,
                "clip_val": 0.005,
                "group": 5,
                "lyr_num": -1,
            },
            "Kuwait": {
                "Lat": 28.563148,
                "Lon": 48.459282,
                "SD": "2017-08-01",
                "ED": "2017-08-15",
                "zoom": 11,
                "clip_val": 0.008,
                "group": 5,
                "lyr_num": -1,
            },
            "Caspian": {
                "Lat": 40.1280,
                "Lon": 50.8818,
                "SD": "2017-12-01",
                "ED": "2017-12-15",
                "zoom": 12,
                "clip_val": 0.006,
                "group": 5,
                "lyr_num": -1,
            },
            "Med": {
                "Lat": 43.2162,
                "Lon": 9.2498,
                "SD": "2018-10-01",
                "ED": "2018-10-31",
                "zoom": 11,
                "clip_val": 0.005,
                "group": 8,
                "lyr_num": 1,
            },
            "Indonesia": {
                "Lat": 5.304552,
                "Lon": 98.093229,
                "SD": "2019-11-15",
                "ED": "2020-02-29",
                "zoom": 11,
                "clip_val": 0.008,
                "group": 8,
                "lyr_num": 3,
            },
        }

    def refresh_area(self, change, initial=False):
        if change is None:
            cur_loc = self.area_selector.value
        else:
            cur_loc = change["new"]

        self.m.center = [self.locations[cur_loc]["Lat"], self.locations[cur_loc]["Lon"]]
        self.m.zoom = self.locations[cur_loc]["zoom"]

        self.ed.value, self.sd.value = (
            datetime.strptime(self.locations[cur_loc]["ED"], "%Y-%m-%d"),
            datetime.strptime(self.locations[cur_loc]["SD"], "%Y-%m-%d"),
        )
        self.SAR_clip.value = self.locations[cur_loc]["clip_val"]
        self.group_days.value = self.locations[cur_loc]["group"]

        if not initial:
            time.sleep(5)
            self.update_layers(None)
