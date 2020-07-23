import numpy as np
from plyfile import PlyData, PlyElement


def write_ply(save_path, point_cloud, text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2],
               point_cloud[i, 3] * 255, point_cloud[i, 4] * 255,
               point_cloud[i, 5] * 255) for i in range(point_cloud.shape[0])]
    vertex = np.array(
        points,
        dtype=[('x', 'float32'), ('y', 'float32'), ('z', 'float32'),
               ('red', 'ubyte'), ('green', 'ubyte'), ('blue', 'ubyte')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)


if __name__ == '__main__':
    data = np.fromfile(
        './demo/sunrgbd_000017.bin', dtype=np.float32).reshape([-1, 6])
    write_ply('./test.ply', data)
