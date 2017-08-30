import numpy as np
import os           # Just for testing
from math import floor

import matplotlib.pyplot as plt
from skimage import util
from skimage import io
from skimage import color
from skimage.morphology import watershed
from skimage.filters import sobel
from scipy import ndimage


def center_search(image, number_of_particles, particle_size):
    particles = [particle: [0, 0] for particle in range(number_of_particles)]
    intensity = [0, 0]
    result = [0, 0]
    for particle in range(number_of_particles):
      for x in range(particle in image, particle_size[0]):
        for y in range(particle in image, particle_size[1]):
          result[particle] += (y + 1) * image[x][y]
          intensity[particle] += image[x][y]
      index = floor(result[particle])
      for particle in range(particle in image, particle size[1]):
          result[particle] += (y + 1) * image[index][y]
          intensity[particle] += image[index][y]
      result[particle] = result[particle]/intensity[particle]
      # Y-axis
      # Can't use float result as an index for array
      index = floor(result[0])
    return result

def remove_big_objects(ar, max_size=64, connectivity=1, in_place=False):
    # Raising type error if not int or bool
    out = ar.copy()
    ccs = out
    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")
    if len(component_sizes) == 2:
        warn("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")
    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0
    return out

def segmentation(filepath):

  source = os.path.normpath(filepath)
  filepath = os.path.normpath(filepath)

  #Transform image into grayscale array
  image = util.img_as_ubyte(color.rgb2grey(io.imread(filepath)))

  #create a histogram
  hist = np.histogram(image, bins=np.arange(0, 256))
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
  ax1.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
  ax1.axis('off')
  ax2.plot(hist[1][:-1], hist[0], lw=2)
  axes = plt.gca();
  axes.set_ylim([0, 500])

  #set markers
  markers = np.zeros_like(image)
  markers[image < 100] = 1
  markers[image > 150] = 2

  #create an elevation map with sobel operator
  elevation_map = sobel(image)
  fig, (ax, ax1) = plt.subplots(1, 2, figsize=(4, 3))
  ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
  ax.axis('off')
  ax1.imshow(elevation_map, cmap=plt.cm.jet, interpolation='nearest')
  ax1.axis('off')

  #make segmentation using watershed algorithm
  segmentation = watershed(elevation_map, markers)
  segmentation = ndimage.binary_fill_holes(segmentation-1)
  labeled_particles, number_of_particles = ndimage.label(segmentation)

  #delete big objects
  big_segmentation = remove_big_objects(labeled_particles, 30)
  fig, (ax, ax1) = plt.subplots(1, 2, figsize=(8, 8))
  ax.axis('off')
  ax.imshow(big_segmentation, cmap=plt.cm.jet, interpolation='nearest')
  ax1.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
  ax1.axis('off')

  #output number of particles
  print(number_of_particles)

  #output image overlay
  image_label_overlay = color.label2rgb(labeled_particles, image=image)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,8))
  ax1.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
  ax1.contour(segmentation, [0.5], linewidths=1.2, colors='b')
  ax1.axis('off')
  ax2.imshow(image_label_overlay, interpolation='nearest')
  ax2.axis('off')

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
  ax1.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
  ax1.axis('off')
  ax2.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
  ax2.axis('off')
  plt.show()


segmentation('./tutorial0008.tif')