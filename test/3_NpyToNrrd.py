import nrrd
import numpy as np
import argparse
from six.moves import cPickle as pickle
import os

image_size = 64

parser = argparse.ArgumentParser()

parser.add_argument('-i', action='store', dest='input_image', help='image .npy')
parser.add_argument('-header', action='store', dest='header', help='image header, .pickle')
options = parser.parse_args()

# get name
input_image = options.input_image
input_header = options.header

save_name = str.split(input_image,  '.')[2]

_, save_name_header = os.path.split(input_image)
save_name_header = str.split(save_name_header,  '.')[0]
save_name_header = str.split(save_name_header, '_prediction')[0]


# read image
image = np.load(input_image)

# read header
with open(input_header, 'rb') as f:
    save = pickle.load(f)
    header = save[save_name_header]
    del save  # hint to help gc free up memory

# save as nrrd
image = image.reshape((image_size, image_size)).astype(np.float32)
nrrd.write('..' + save_name + '.nrrd', image, header)
