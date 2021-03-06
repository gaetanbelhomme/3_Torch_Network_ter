import argparse
import numpy as np
import nrrd
import os
from six.moves import cPickle as pickle

image_size = 64
path = '../Data/test/'

parser = argparse.ArgumentParser()

parser.add_argument('-i', action='store', dest='inputImage', help='image .nrrd')
parser.add_argument('-o', action='store', dest='path', help='output path')

options = parser.parse_args()

# get name
output_path = options.path
input_image = options.inputImage
_, name_save = os.path.split(input_image)
name_save = str.split(name_save, '.')[0]

# read image
input_image, input_header = nrrd.read(input_image)

#input_image = (input_image.astype(float) - 255.5 / 2) / 255.0
#input_image = (input_image.astype(float)) / 65536.0

input_image = input_image.reshape((1,image_size, image_size)).astype(np.float32)

# save as npy
print output_path
np.save(output_path + name_save + '.npy', input_image)

pickle_file = output_path + name_save + '_header.pickle'

save = {name_save: input_header}
try:
    f = open(pickle_file, 'wb')
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

