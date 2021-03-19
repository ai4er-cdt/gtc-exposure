import numpy as np
import descarteslabs as dl
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image

def generate_sentinel_training_images(geometry,
                                      area, #eg. 'Jamaca'
                                      tile_size= 512,
                                      start_datetime="2014-01-01",
                                      end_datetime="2020-01-01",
                                      cloud_fraction=0.01
                                     ):
    
    #use descartes API to get Sentinel image
    scenes, geoctx = dl.scenes.search(geometry,
                                      products=["sentinel-2:L1C"],
                                      start_datetime= start_datetime,
                                      end_datetime = end_datetime,
                                      cloud_fraction = cloud_fraction)
    
    #creates image stack using RGB Bands
    ndarray_stack = scenes.stack("red green blue", geoctx.assign())
    
    #for each of the images
    image_stack = []
    for img in ndarray_stack:
        tiles= []
        #slice the image into tiles 
        for y in range(tile_size, img.shape[2], tile_size):
            for x in range(tile_size, img.shape[1], tile_size):
                tile = (img[:,x:x+tile_size, y:y+tile_size])
                #this filters edge images that are not the correct shape
                if tile.shape == (3, tile_size, tile_size):
                    tiles.append(tile)             
        image_stack.append(tiles)
        
    #convert nested list to array
    no_scenes = len(image_stack)
    no_tiles_per_scene = len(image_stack[0])
    image_array  = np.zeros([no_scenes, no_tiles_per_scene, 3, tile_size, tile_size])
    for i in range(no_scenes):
        for j in range(no_tiles_per_scene):
            image_array[i, j] = image_stack[i][j]
      
    #take compoite image as average of scenes
    composite_images = np.zeros(image_array[0].shape)
    for i in range(image_array.shape[1]):
        composite_image = np.ma.median(image_array[:,i], axis=0)
        composite_images[i] = composite_image
        
        # reshape from (3, x, y) to (x, y, 3)
        reshape_image = np.zeros((tile_size,tile_size,3))
        reshape_image[:,:,0] = composite_image[0]
        reshape_image[:,:,1] = composite_image[1]
        reshape_image[:,:,2] = composite_image[2]
        #scale values to [0, 255]
        #avoid divide by 0 error:
        if np.max(reshape_image)!=0:
            reshape_image = (reshape_image/np.max(reshape_image)*255).astype(np.uint8)
            #save images as jpeg
            Image.fromarray(reshape_image).save("train_{}_{}.jpeg".format(area, i))
        
    return composite_images