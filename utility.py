# import png
import numpy as np
import scipy.misc
import math
from scipy import signal        # for convolutions
from scipy import ndimage       # for n-dimensional convolution
from scipy import interpolate


class special_function:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)  # [w, h]

    def bilateral_filter(self, edge):
        '''双边滤波，保持边缘
        空间上做高斯平滑，但抑制边缘的平滑
        '''
        width, height = self.data.shape

        # 空间域，控制上离中心像素多远的邻居还会被拿来平均——越大就越“看得远”、平滑得越狠
        sigma_spatial = min(height, width) / 10.0

        edge_min = np.min(edge)
        edge_max = np.max(edge)
        edge_delta = edge_max - edge_min

        # 值域，控制差多少值算同一类，越大就允许差很多也能被平均
        sigma_range = edge_delta / 10.0
        sampling_range = sigma_range
        sampling_spatial = sigma_spatial

        derived_sigma_spatial = sigma_spatial / sampling_spatial
        derived_sigma_range = sigma_range / sampling_range

        padding_xy = np.floor(2. * derived_sigma_spatial) + 1.
        padding_z = np.floor(2. * derived_sigma_range) + 1.

        # 图像->3d点云，x, y, z分别是图像的列、行、像素值
        # 考虑图像
        # 1  1
        # 0  1
        # 
        # A small ASCII isometric sketch (parallelogram 3D view):
        #
        #            z (range bins)
        #            ^
        #            |
        #            1----1
        #           /|   /|
        #          .----1 |
        #          | |--| +----> y
        #          |/   |/
        #          0----.
        #         /
        #        v
        #        x
        #

        downsample_width = np.floor((width - 1) / sampling_spatial) + 1 + 2 * padding_xy
        downsample_height = np.floor((height - 1) / sampling_spatial) + 1 + 2 * padding_xy
        downsample_depth = np.floor(edge_delta / sampling_range) + 1 + 2 * padding_z

        yy, xx = np.meshgrid(np.arange(width), np.arange(height))
        

        return

class filter:
    @staticmethod
    def gaussian(kernel_size=3, sigma=1):
        '''generate a gaussian kernel
        Args:
            kernel_size: size of the kernel, int
            sigma: standard deviation of the gaussian distribution, float
        Returns:
            kernel: gaussian kernel, numpy array
        '''
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / np.sum(kernel)  # 归一化
    
    @staticmethod
    def sobel(kernel_size=3):
        a = np.array([[1,2,1]])  # [1, 3]
        b = np.array([[1,0,-1]])  # [1, 3]
        Sx = 0.25 * np.dot(a.T, b)  # [3, 3]

        if kernel_size > 3:
            n = int((kernel_size - 3) / 2)
            for i in range(n):
                c = np.array([[1,2,1]])  # [1, 3]
                d = np.array([[1,2,1]])  # [1, 3]
                Sx = (1./16.) * signal.convolve2d(np.dot(c.T, d), Sx, mode="full")  # [5, 5], [7, 7], ...

        Sy = Sx.T  # [3, 3], [5, 5], [7, 7], ...

        return Sx, Sy
    

class edge_detection:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)  # [w, h]

    def sobel(self, kernel_size=3, type="all", threshold=0.0, clip_range=[0, 65535]):
        '''threshold should be [0, 1]'''
        Sx, Sy = filter.sobel(kernel_size=kernel_size)

        if np.dim(self.data) > 2:
            Gx = np.empty(self.data.shape, dtype=np.float32)
            Gy = np.empty(self.data.shape, dtype=np.float32)

            for i in range(self.data.shape[2]):
                Gx[:, :, i] = signal.convolve2d(self.data[:, :, i], Sx, mode='same', boundary='symm')
                Gy[:, :, i] = signal.convolve2d(self.data[:, :, i], Sy, mode='same', boundary='symm')

        # 得到梯度图像
        elif np.dim(self.data) == 2:
            Gx = signal.convolve2d(self.data, Sx, mode='same', boundary='symm')
            Gy = signal.convolve2d(self.data, Sy, mode='same', boundary='symm')
        else:
            raise ValueError("Input data must be 2D or 3D array")
        
        G = np.sqrt(Gx**2 + Gy**2)

        if (type == "gradient_magnitude"):
            return G
        
        theta = np.arctan(np.divide(Gy, Gx)) * 180. / np.pi

        if (type == "gradient_magnitude_and_angle"):
            return G, theta
        
        # 根据梯度图像和阈值得到边缘图像
        threshold = threshold * clip_range[1]
        edge = np.zeros_like(G, dtype=np.int)
        mask = G > threshold
        edge[mask] = 1

        if (type == "edge_map"):
            return edge
        


    