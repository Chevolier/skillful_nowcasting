import bz2
import struct
import zipfile
import numpy as np


def read_bin(path):
    with open(r'data/Z_RADA_C_BABJ_20231109020616_P_DOR_ACHN_OHP06_20231109_020000.bin', 'rb+') as f:
        bt = f.read()
        edge_s = struct.unpack('i', bt[124:128])[0] / 1000
        edge_w = struct.unpack('i', bt[128:132])[0] / 1000
        edge_n = struct.unpack('i', bt[132:136])[0] / 1000
        edge_e = struct.unpack('i', bt[136:140])[0] / 1000

        nX = struct.unpack('i', bt[148:152])[0]
        nY = struct.unpack('i', bt[152:156])[0]

        max_lon = edge_e  # max(lons)
        min_lon = edge_w  # min(lons)
        max_lat = edge_n  # max(lats)
        min_lat = edge_s  # min(lats)

        s = bz2.decompress(bt[256:])
        print(len(s) / 2 / 4200)
        n = []
        for i in range(0, nY):
            inner = []
            for j in range(0, nX):
                a = struct.unpack('h', s[((i * nX * 2) + j * 2):((i * nX * 2) + j * 2 + 2)])[0]
                inner.append(a / 10.0)
            n.append(inner)

        return n