# import packages
import descarteslabs.workflows as wf

import ipyleaflet
from ipyleaflet import GeoData
import ipywidgets as widgets

import numpy as np

import affine
from rasterio.features import shapes

from skimage import morphology
import scipy.ndimage.morphology as sm
import math

import geopandas as gpd

# keep logging quiet
import logging

logging.getLogger().setLevel(logging.INFO)
logging.captureWarnings(True)


def wgs_to_epsg(lat, lon):
    """
    Get the epsg code from a (lat, lon) location
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band
    else:
        epsg_code = "327" + utm_band
    return epsg_code


class FieldMap:
    def __init__(self, map_widget):
        """
        Initialize map, data attributes, controls, logs, and events
        """
        self.m = map_widget
        self.fields = None

        # Set up draw control for rectangle
        self.draw_control = ipyleaflet.DrawControl(
            edit=False, remove=False, circlemarker={}, polyline={}, polygon={}
        )
        self.draw_control.rectangle = {
            "shapeOptions": {
                "fillColor": "#ffcc00",
                "color": "#ffcc00",
                "fillOpacity": 0.2,
            }
        }
        self.m.add_control(self.draw_control)

        # Add layer control
        self.layer_control = ipyleaflet.LayersControl(position="topright")
        self.m.add_control(self.layer_control)

        # Set up button for exporting geojson of vectorized fields
        self.export_geojson_control = ipyleaflet.WidgetControl(
            widget=widgets.Button(
                description="Export GeoJSON",
                disabled=False,
                button_style="warning",
                tooltip="Export vectorized field polygons to GeoJSON file",
            ),
            position="bottomright",
        )

        # OUTPUTS
        self.output_log = widgets.Output()

        # EVENTS
        self.draw_control.on_draw(self.update_fields)
        self.export_geojson_control.widget.on_click(self.export_geojson)

    def update_fields(self, *args, geo_json={}, **kwargs):
        """
        Clean up fields within a selected geojson,
         and display as polygons on a map
        """
        # Remove existing polygons - we'll generate new ones
        if hasattr(self, "fields_df"):
            self.m.remove_layer(self.geo)

        self.crs = f"EPSG:{wgs_to_epsg(*self.m.center)}"

        self.vector_resolution = 10.0

        # Compute affine transform, used for vectorization
        self.draw_df = gpd.GeoDataFrame.from_features([geo_json])
        self.draw_df.crs = {"init": "EPSG:4326"}
        self.draw_df = self.draw_df.to_crs({"init": self.crs})
        utm_bounds = self.draw_df.loc[0].geometry.bounds
        geotransform = (
            utm_bounds[0],
            self.vector_resolution,
            0.0,
            utm_bounds[3],
            0.0,
            -self.vector_resolution,
        )
        af = affine.Affine.from_gdal(*geotransform)

        # Create geocontext from selected rectangular region
        self.geoctx = wf.GeoContext(
            geometry=geo_json["geometry"],
            resolution=self.vector_resolution,
            crs=self.crs,
            bounds_crs="EPSG:4326",
        )

        # Get fields array from the "Initial Fields" layer on the map
        fields_array = self.fields.compute(self.geoctx, progress_bar=False)
        self.fields_array = np.squeeze(fields_array.ndarray)

        # Clean up segmented fields
        self.fields_clean = self.clean_fields()

        # Get polygons of fields from the cleaned fields
        geom_iter = shapes(self.fields_clean, mask=None, transform=af)

        self.geoms = [
            {"properties": {"raster_val": v}, "geometry": s, "type": "feature"}
            for s, v in geom_iter
        ]
        self.fields_df = gpd.GeoDataFrame.from_features(self.geoms[:-1])
        self.fields_df.crs = {"init": self.crs}
        self.fields_df["area [Ha]"] = np.round(self.fields_df.area * 0.0001, 2)
        self.fields_df = self.fields_df.to_crs({"init": "EPSG:4326"})

        geo = GeoData(
            geo_dataframe=self.fields_df,
            style={
                "color": "#ffcc00",
                "opacity": 1,
                "weight": 1.9,
                "fillColor": "#ffcc00",
                "fillOpactiy": 0.3,
            },
            hover_style={"fillColor": "#ffcc00", "fillOpacity": 0.5},
            name="Vectorized Fields",
        )

        def hover_handler(event=None, feature=None, id=None, properties=None):
            """
            Show a field's area when the user is hovered over it
            """
            self.label.value = f"Field Area: {str(properties['area [Ha]'])} Ha"

        # Add controls if they don't already exist
        if not hasattr(self, "label"):
            # geojson button control
            self.m.add_control(self.export_geojson_control)
            # label control
            self.label = widgets.Label()
            self.label_control = ipyleaflet.WidgetControl(
                widget=self.label, position="bottomright"
            )
            self.m.add_control(self.label_control)

        # Enable hover handler on the polygon layer
        geo.on_hover(hover_handler)

        # Add polygons layer to the map
        self.geo = geo
        self.m.add_layer(self.geo)

        # Remove the last drawn rectangle
        self.draw_control.clear()

    def export_geojson(self, *args, **kwargs):
        """
        Export polygons to a GeoJSON
        """
        self.fields_df.to_file("my_fields.geojson", driver="GeoJSON")

    def sieve_bool(self, arr, min_size=1, connectivity=1, max_size=None):
        """
        Remove small polygons from a boolean array

        Parameters
        ----------
        arr : boolean array
            array to fill
        min_size : int
            minimum polygon size to keep in array
        connectivity : int
            pixel connectivity
        max_size: int
            maximum polygon size to keep in array

        Returns
        -------
        filled : boolean array
            input array with small polygons filled
        """
        # initialize mask containing holes
        mask = np.zeros_like(arr, dtype=bool)

        # create binary array for each class
        # and find contiguous patches > min_size
        for cls in [True, False]:
            binary = np.where(arr == cls, True, False)
            m = morphology.remove_small_objects(
                binary.astype(bool),
                min_size=min_size,
                connectivity=connectivity,
                in_place=False,
            )
            mask = np.where(m == True, True, mask) # noqa

        # fill small objects from mask
        filled = np.where(mask == True, ~arr, arr) # noqa

        # fill small interior holes
        filled = sm.binary_fill_holes(filled, structure=np.ones((7, 7))).astype(int)

        return filled

    def clean_fields(self):
        """
        Clean up fields by removing small objects and
         applying erosion and dilation operators
        """
        # remove small objects
        fields = self.sieve_bool(
            ~(self.fields_array > 0), min_size=100, max_size=None, connectivity=1
        )

        # define structure to connect diagonally connected pixels
        struct = sm.generate_binary_structure(2, 2)

        # erosion and dilation for cleaning
        fields = sm.binary_erosion(fields, structure=struct, iterations=3)
        fields = sm.binary_dilation(fields, structure=struct, iterations=1)

        # remove small objects
        fields = self.sieve_bool(
            ~(fields > 0), min_size=100, max_size=None, connectivity=1
        )
        fields = fields.astype(np.uint8)

        return fields
