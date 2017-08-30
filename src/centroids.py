import os           # Just for testing
from math import floor
from skimage import util
from skimage import io
from skimage import color
from skimage import data
import numpy as np

class Imageprocess(object):
  def __init__(self, filepath):
    filepath = os.path.normpath(filepath)
    self.filepath = filepath


  def count_centroids(self):
    image = util.img_as_ubyte(color.rgb2grey(io.imread(self.filepath)))
    image_size = np.shape(image)
    intensity = [0, 0]
    result = [0, 0]
    # Weighted centroid coordinates
    # X-axis
    for i in range(0, image_size[0]):
      for j in range(0, image_size[1]):
        result[0] += (j + 1) * image[i][j]
        intensity[0] += image[i][j]
    result[0] = result[0]/intensity[0]
    # Y-axis
    # Can't use float result as an index for array
    index = floor(result[0])
    for y in range(0, image_size[1]):
      result[1] += (y + 1) * image[index][y]
      intensity[1] += image[index][y]
    result[1] = result[1]/intensity[1]
    return result


img = Imageprocess('./img2.tif')

print(img.count_centroids())