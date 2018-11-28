import numpy as np

class YV12Reader:
    def __init__(self, size, filename):
        self.width, self.height = size
        width, height = size
        self.filename = filename
        num_sub_pixels = width * height // 4
        self.dtype = np.dtype([
            ('Y', np.uint8, (self.height, self.width)),
            ('U', np.uint8, (num_sub_pixels,)),
            ('V', np.uint8, (num_sub_pixels,))
        ])

    def read_raw_frames(self):
        return np.fromfile(self.filename, dtype=self.dtype)

    @staticmethod
    def _convert_frame(raw_data):
        height, width = raw_data['Y'].shape

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                uv_offset = (i // 2) * (width // 2) + (j // 2)
                frame[i][j] = np.array([
                    raw_data['Y'][i][j],
                    raw_data['U'][uv_offset],
                    raw_data['V'][uv_offset]
                ])
        return frame

    def read_frames(self):
        return (YV12Reader._convert_frame(raw_data)
                for raw_data in self.read_raw_frames())


def yuv2rgb(data):
    def _clip(x):
        if x < 0:
            return 0
        if x > 255:
            return 255
        return x

    rgb = np.zeros(data.shape, dtype=np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            y, v, u = data[i][j]
            c = y - 16
            d = u - 128
            e = v - 128
            rgb[i][j] = np.array((
                _clip((298 * c           + 409 * e + 128) >> 8),
                _clip((298 * c - 100 * d - 208 * e + 128) >> 8),
                _clip((298 * c + 516 * d           + 128) >> 8)
            ))
    return rgb
