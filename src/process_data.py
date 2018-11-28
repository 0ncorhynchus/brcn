import cv2
import numpy as np
import os
from yuv import *
import tensorflow as tf
from example import encode

video_size = (176, 144)

spatial_size, spatial_stride = (44, 14)
temporal_size, temporal_stride = (22, 8)

blur_size = 2
scale = 2

def generate_low_resolution(image):
    blur = cv2.GaussianBlur(image, (5, 5), blur_size)
    size = (blur.shape[1], blur.shape[0])
    small_size = (size[0] // scale, size[1] // scale)
    small = cv2.resize(blur, small_size, interpolation = cv2.INTER_CUBIC)
    low_reso = cv2.resize(small, size, interpolation = cv2.INTER_CUBIC)
    return low_reso


def main(filename):
    reader = YV12Reader(video_size, filename)

    print('Loading frames from "{}".'.format(filename))

    high_reso_frames = reader.read_raw_frames()['Y']
    low_reso_frames = np.array(list(map(generate_low_resolution, high_reso_frames)))

    num_frames = len(high_reso_frames)

    basename = os.path.basename(filename)
    outfile = '{}.tfrecord'.format(os.path.splitext(basename)[0])

    print('Writing training data to "{}".'.format(outfile))

    with tf.python_io.TFRecordWriter(outfile) as writer:
        for seq in range(0, num_frames-temporal_size+1, temporal_stride):
            for i in range(0, video_size[0]-spatial_size+1, spatial_stride):
                for j in range(0, video_size[1]-spatial_size+1, spatial_stride):
                    hr_sample = high_reso_frames[
                        seq:seq+temporal_size,
                        j:j+spatial_size,
                        i:i+spatial_size
                    ]
                    lr_sample = low_reso_frames[
                        seq:seq+temporal_size,
                        j:j+spatial_size,
                        i:i+spatial_size
                    ]
                    writer.write(encode(lr_sample, hr_sample).SerializeToString());


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
