import idx2numpy
import numpy as np

file = 'train-images-idx3-ubyte'
arr = idx2numpy.convert_from_file(file)

cv.imshow("Image", arr[4])