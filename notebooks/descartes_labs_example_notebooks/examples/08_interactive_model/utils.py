import descarteslabs as dl
import descarteslabs.workflows as wf
import numpy as np
import datetime
import colorsys
import psutil
import multiprocessing as mp

from ipyleaflet import DrawControl, WidgetControl, Polygon, LayersControl
from ipywidgets import Button, Layout, IntProgress
from skimage import measure
from skimage.morphology import binary_opening
import shapely.geometry as geo
from shapely.ops import unary_union

import tensorflow as tf

MAX_MODELS = 5
LAYER_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


def rowcol_to_lonlat(tile, contours):
    # Convert xy coordinates to lon lat. This is not the best way to do this,
    # but should work well enough for smaller tiles.
    # For a more accurate method use the descarteslabs_tile module.

    # Find the upperleft-most point in our geometry, upper-right most, and lower-left most
    ul, ur, ll = None, None, None
    for point in zip(*tile.geometry.exterior.xy):
        if ul is None or point[1] - point[0] > ul[1] - ul[0]:
            ul = np.array(point)
        if ur is None or point[1] + point[0] > ur[1] + ur[0]:
            ur = np.array(point)
        if ll is None or point[0] + point[1] < ll[0] + ll[1]:
            ll = np.array(point)

    # Get column and row vectors for lon,lat displacement
    tile_extent = tile.tilesize + 2 * tile.pad
    v_col = (ur - ul) / tile_extent
    v_row = (ll - ul) / tile_extent
    min_lon, max_lat = ul

    for contour in contours:
        coords = []
        for row, col in contour:
            lon, lat = np.array((min_lon, max_lat)) + col * v_col + row * v_row
            coords.append((lat, lon))
        yield coords


class LFUCache(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.images = {}
        self.frequencies = {}
        self.min_frequency = 0
        self.count = 0

    def _increment_frequency(self, key, f):
        f_group = self.frequencies[f]
        f_group.remove(key)

        f_group = self.frequencies.get(f + 1, [])
        f_group.insert(0, key)
        self.frequencies[f + 1] = f_group

        if len(self.frequencies[f]) == 0 and f == self.min_frequency:
            self.min_frequency = f + 1

    def __getitem__(self, key):
        if key not in self.images:
            return []

        val, f = self.images[key]
        self.images[key] = (val, f + 1)

        self._increment_frequency(key, f)

        return val

    def __setitem__(self, key, value):
        if key in self.images:
            val, f = self.images[key]
            self.images[key] = (value, f + 1)

            self._increment_frequency(key, f)

            return

        if self.count == self.max_size:
            f = self.frequencies[self.min_frequency]
            k = f.pop()

            del self.images[k]
            self.count -= 1

        self.images[key] = (value, 1)
        self.count += 1

        f = self.frequencies.get(1, [])
        f.insert(0, key)
        self.frequencies[1] = f
        self.min_frequency = 1

        return

    def __contains__(self, key):
        return key in self.images


class Model(object):
    def __init__(
        self,
        model,
        name,
        checkpoint_path=None,
        checkpoint=None,
        n_classes=1,
        label_colors=None,
        pre_process_fn=None,
        post_process_fn=None,
    ):
        self.model = model
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.checkpoint = checkpoint
        self.n_classes = n_classes
        self.label_colors = label_colors
        self.pre_process_fn = pre_process_fn
        self.post_process_fn = post_process_fn


class Deploy(object):
    def __init__(
        self,
        map_widget,
        draw_control=None,
        product=None,
        resolution=None,
        tilesize=256,
        bands=None,
        start_datetime="2016-01-01",
        end_datetime="2019-12-31",
        model=None,
        model_name="model_1",
        checkpoint_path=None,
        checkpoint=None,
        n_classes=1,
        label_colors=None,
        max_batch_size=32,
        pre_process_fn=None,
        post_process_fn=None,
        use_cache=True,
    ):
        self.m = map_widget
        if not draw_control:
            self.draw_control = DrawControl()
            self.draw_control.edit = False
            self.draw_control.remove = False
            self.draw_control.circle = {}
            self.draw_control.circlemarker = {}
            self.draw_control.polyline = {}
            self.draw_control.rectangle = {
                "shapeOptions": {
                    "fillColor": "#000000",
                    "color": "#fca45d",
                    "fillOpacity": 0.0,
                }
            }
            self.draw_control.polygon = {}
        else:
            self.draw_control = draw_control

        self.layers_control = LayersControl(position="topright")

        self.int_progress = IntProgress(value=0, min=0, max=100)

        self.widget_control = WidgetControl(
            widget=self.int_progress,
            max_width=400,
            max_height=30,
            position="bottomleft",
        )

        self.remove_polygons_button = Button(
            layout=Layout(width="37px"), description="\u274C", tooltip="Remove polygons"
        )

        self.remove_polygons_button_control = WidgetControl(
            widget=self.remove_polygons_button,
            max_width=50,
            max_height=50,
            position="topleft",
        )

        self.remove_polygons_button.on_click(self.remove_polygons)

        self.m.add_control(self.draw_control)

        self.m.add_control(self.layers_control)

        self.m.add_control(self.remove_polygons_button_control)

        # self.m.add_control(self.widget_control)

        self.product = product
        self.resolution = resolution
        self.tilesize = tilesize
        self.bands = bands
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.models = []
        self.max_batch_size = max_batch_size

        self.use_cache = use_cache
        if self.use_cache:
            # Determine the cache size based on the available memory
            # and the image size. Choose 10% of the available memory
            available_memory = 0.1 * psutil.virtual_memory().available
            cache_size = int(
                available_memory / (self.tilesize ** 2 * len(self.bands) * 32) * 8
            )
            print(
                "Using a cache size of {0:.0f} ({1:.0f} MB)".format(
                    cache_size, available_memory / 1024 ** 2
                )
            )
            self.image_cache = LFUCache(cache_size)

        self.times = {}

        self.pool = mp.Pool(2 * self.max_batch_size)
        self.draw_control.on_draw(self.compute)

        if model is not None:
            self.add_model(
                model,
                model_name,
                checkpoint_path,
                checkpoint,
                n_classes,
                label_colors,
                pre_process_fn,
                post_process_fn,
                compute=False,
            )

    @staticmethod
    def load_dltile(args):
        # To be run in multiprocessing
        dltile, scene_ids, bands = args
        arr, meta = dl.raster.ndarray(
            scene_ids, bands=bands, dltile=dltile.properties.key
        )
        return dltile, arr

    @wf.map.output_log.capture(clear_output=True)
    def compute(self, *argv, new_model=False, **kwargs):
        last_draw = self.draw_control.last_draw
        if not last_draw["geometry"]:
            # There is nothing to compute
            return
        dltiles = dl.raster.dltiles_from_shape(
            self.resolution, self.tilesize, 0, last_draw
        )["features"]

        self.m.add_control(self.widget_control)
        self.int_progress.value = 0
        self.int_progress.max = len(dltiles) + 2

        self.times = {}
        start_time = datetime.datetime.now()
        start_time_total = datetime.datetime.now()

        # Get all scenes over dltiles at once
        scenes_shape = unary_union([geo.shape(dltile.geometry) for dltile in dltiles])
        scenes, ctx = dl.scenes.search(
            scenes_shape,
            products=self.product,
            start_datetime=self.start_datetime,
            end_datetime=self.end_datetime,
        )
        scene_ids = [scene.properties.id for scene in scenes]

        self.update_compute_times("scenes_call", start_time)

        start_idx = (len(self.models) - 1) if new_model else 0

        model_polygons = [[list()] for _ in range(len(self.models))]

        # Split the dltiles to get from the image cache
        cached_dltiles = []
        args = []
        for dltile in dltiles:
            if dltile.properties.key in self.image_cache and self.use_cache:
                cached_dltiles.append(dltile)
            else:
                args.append((dltile, scene_ids, self.bands))

        n_dltiles = len(dltiles)
        image_list = []
        tile_list = []

        start_time = datetime.datetime.now()

        # Fill up the batch with images from the image cache
        for image_i, dltile in enumerate(cached_dltiles):
            self.int_progress.value += 1

            image_list.append(self.image_cache[dltile.properties.key])
            tile_list.append(dltile)

            if len(image_list) < self.max_batch_size and image_i + 1 < n_dltiles:
                continue

            start_time_inference = datetime.datetime.now()
            model_polygons = self.run_batch_inference(
                image_list, tile_list, model_polygons, start_idx
            )
            self.update_compute_times("model_inference", start_time_inference)

            del image_list[:]
            del tile_list[:]

        self.update_compute_times("model_inference_cache", start_time)
        start_time = datetime.datetime.now()

        n_dltiles = len(args)

        # Get the remaining raster images
        for image_i, (dltile, image) in enumerate(
            self.pool.imap_unordered(self.load_dltile, args)
        ):
            self.int_progress.value += 1

            image_list.append(image)
            tile_list.append(dltile)

            self.image_cache[dltile.properties.key] = image

            if len(image_list) < self.max_batch_size and image_i + 1 < n_dltiles:
                # Not enough elements for a batch yet, and also not the last element
                continue

            start_time_inference = datetime.datetime.now()
            model_polygons = self.run_batch_inference(
                image_list, tile_list, model_polygons, start_idx
            )
            self.update_compute_times("model_inference", start_time_inference)

            del image_list[:]
            del tile_list[:]

        self.update_compute_times("model_inference_raster", start_time)
        start_time = datetime.datetime.now()

        self.int_progress.value += 1

        for model_i, (polygons, model) in enumerate(zip(model_polygons, self.models)):
            if len(polygons):
                # Create one multipolygon per model and output layer
                # If the colors are not defined in the model create the colors
                # from the base color
                if model.label_colors is None:
                    model.label_colors = [LAYER_COLORS[model_i % len(LAYER_COLORS)]]

                if not len(model.label_colors) == len(polygons):
                    base_color = model.label_colors[0]
                    for i in range(1, len(polygons)):
                        hsv = list(colorsys.rgb_to_hsv(*base_color))
                        hsv[0] += 0.2
                        model.label_colors.append(colorsys.hsv_to_rgb(*hsv))

                # Make sure the layer colors are in string format
                model.label_colors = self.convert_to_string_colors(model.label_colors)

                for polygon_i, polygon in enumerate(polygons):
                    if len(polygon) == 0:
                        continue

                    polygon = unary_union(
                        [
                            geo.shape({"type": "polygon", "coordinates": [geometry]})
                            for geometry in polygon
                        ]
                    )

                    if type(polygon) == geo.multipolygon.MultiPolygon:
                        polygon = [list(p.exterior.coords) for p in polygon]
                    elif type(polygon) == geo.GeometryCollection:
                        polygon = [
                            list(p.exterior.coords)
                            for p in polygon
                            if type(p) == geo.polygon.Polygon
                        ]
                    else:
                        polygon = list(polygon.exterior.coords)

                    multi_polygon = Polygon(
                        name=model.name + "_layer{}".format(polygon_i),
                        locations=polygon,
                        color=model.label_colors[polygon_i],
                        fill_color=model.label_colors[polygon_i],
                        weight=0,
                        fill_opacity=0.8,
                    )

                    self.m.add_layer(multi_polygon)

        self.update_compute_times("polygon_draw", start_time)
        self.update_compute_times("total_time", start_time_total)

        self.int_progress.value = 0

        self.m.remove_control(self.widget_control)

    def run_batch_inference(self, image_list, tile_list, model_polygons, start_idx=0):
        images = np.array(image_list).astype(np.float32)

        for model_i, model in enumerate(self.models[start_idx:]):
            predictions = model.model(model.pre_process_fn(images))
            results = model.post_process_fn(predictions, tile_list)
            model_polygons[model_i + start_idx] = self.append_polygons(
                model_polygons[model_i + start_idx], results
            )

        return model_polygons

    def append_polygons(self, polygons, results):
        if not len(polygons) == len(results):
            polygons = [list() for _ in range(len(results))]

        for i, l in enumerate(polygons):
            l.extend(results[i])

        return polygons

    def convert_to_string_colors(self, colors):
        return [
            c
            if isinstance(c, str)
            else "#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2]))
            for c in colors
        ]

    def update_compute_times(self, key, start_time):
        compute_time = self.times.get(key, [])
        compute_time.append(datetime.datetime.now() - start_time)
        self.times[key] = compute_time

    def print_compute_times(self):
        for k, v in self.times.items():
            print("{}: {}s".format(k, np.mean(v)))

    def add_model(
        self,
        model,
        model_name=None,
        checkpoint_path=None,
        checkpoint=None,
        n_classes=1,
        label_colors=None,
        pre_process_fn=None,
        post_process_fn=None,
        compute=True,
    ):
        if model_name is None:
            model_name = "model_{}".format(len(self.models) + 1)

        if pre_process_fn is None:
            pre_process_fn = self.default_pre_process_fn

        if post_process_fn is None:
            post_process_fn = self.default_post_process_fn

        self.models.append(
            Model(
                model,
                model_name,
                checkpoint_path,
                checkpoint,
                n_classes,
                label_colors,
                pre_process_fn,
                post_process_fn,
            )
        )

        # Restore latest model checkpoint
        if checkpoint_path is not None:
            ckpt = tf.train.Checkpoint(
                model=self.models[-1].model, optimizer=tf.keras.optimizers.Adam(1e-4)
            )
            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, 5)

            if checkpoint is not None:
                restore_ckpt = checkpoint
            else:
                restore_ckpt = ckpt_manager.latest_checkpoint

            if restore_ckpt:
                print("Loading checkpoint {}".format(restore_ckpt))
                ckpt.restore(restore_ckpt)
        elif checkpoint is None:
            print("WARNING: did not load any checkpoint into model")

        if compute:
            self.compute(new_model=True)

    def remove_model(self, model_name):
        for idx, m in enumerate(self.models):
            if m.name == model_name:
                del self.models[idx]

                break

        for l in self.m.layers:
            if model_name in l.name:
                self.m.remove_layer(l)

                break

    def get_model(self, model_name):
        for m in self.models:
            if m.name == model_name:
                return m.model

        return None

    def remove_polygons(self, *argv, **kwargs):
        for l in self.m.layers:
            if isinstance(l, Polygon):
                self.m.remove_layer(l)

        self.draw_control.clear()

    def default_pre_process_fn(self, images):
        """ The default pre-processing function just returns the images. """
        return images

    def default_post_process_fn(self, output, meta):
        """ Default post-processing function which applies a threshold
        of 0.9 to the output map and finds polygons. """
        polygons = [list() for _ in range(output[0].shape[-1])]

        for out, m in zip(output, meta):
            for i, layer in enumerate(np.rollaxis(np.array(out), axis=-1)):
                layer = np.pad(np.squeeze(layer), [[1, 1], [1, 1]])
                layer = (layer > 0.9).astype(np.int32)
                layer = binary_opening(layer, np.ones(shape=(5, 5)))

                contours = np.array(measure.find_contours(layer, 0.8)) - 1
                dltile = dl.scenes.DLTile.from_key(m.properties.key)
                polygons[i].extend(rowcol_to_lonlat(dltile, contours))

        return polygons
