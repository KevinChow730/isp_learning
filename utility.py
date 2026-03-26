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
        sigma_spatial = min(height, width) / 16

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

        '''
        # 图像->3d点云，x, y, z分别是图像的列、行、像素值
        # 考虑图像
        # 1  1
        # 0  1
        #            z
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
        '''
        downsample_width = np.floor((width - 1) / sampling_spatial) + 1 + 2 * padding_xy
        downsample_height = np.floor((height - 1) / sampling_spatial) + 1 + 2 * padding_xy
        downsample_depth = np.floor(edge_delta / sampling_range) + 1 + 2 * padding_z

        yy, xx = np.meshgrid(np.arange(width), np.arange(height))  # 分别存储所有点的列坐标和行坐标，xx[i, j]是第i行第j列的列坐标，yy[i, j]是第i行第j列的行坐标

        # 把原图像素点投影到3d网格上，网格的x、y、z分别是图像的列、行、像素值
        grid_xx = np.uint16(np.round(xx / sampling_spatial) + padding_xy + 1)
        grid_yy = np.uint16(np.round(yy / sampling_spatial) + padding_xy + 1)
        grid_zz = np.uint16(np.round((edge - edge_min) / sampling_range) + padding_z + 1)

        grid_data = np.zeros((downsample_width, downsample_height, downsample_depth), dtype=np.float32)
        grid_weight = np.zeros((downsample_width, downsample_height, downsample_depth), dtype=np.float32)

        for i in range(width):
            for j in range(height):
                x = grid_xx[i, j]
                y = grid_yy[i, j]
                z = grid_zz[i, j]

                grid_data[x, y, z] += self.data[i, j]  # 多个像素被投影到同一个网格上，网格内的值是所有像素值的平均值
                grid_weight[x, y, z] += 1.0  # 统计每个网格上有多少像素被投影到这个网格上
        
        # 定义一个三维高斯核，分别在空间域和值域上做高斯平滑
        kernel_width = 2. * derived_sigma_spatial + 1.
        kernel_height = kernel_width
        kernel_depth = 2. * derived_sigma_range + 1.

        kernel_x, kernel_y, kernel_z = np.meshgrid(np.arange(kernel_width), np.arange(kernel_height), np.arange(kernel_depth))
        kernel_x -= np.floor(kernel_width / 2.)
        kernel_y -= np.floor(kernel_height / 2.)
        kernel_z -= np.floor(kernel_depth / 2.)

        kernel_r_squared = (kernel_x**2 + kernel_y**2) / derived_sigma_spatial**2\
            + (kernel_z**2) / derived_sigma_range**2
        kernel = np.exp(-0.5 * kernel_r_squared)

        blurred_grid_data = ndimage.convolve(grid_data, kernel, mode='reflect')
        blurred_grid_weight = ndimage.convolve(grid_weight, kernel, mode='reflect')

        # divide
        blurred_grid_weight = np.asarray(blurred_grid_weight)
        mask = blurred_grid_weight == 0
        blurred_grid_weight[mask] = -2.  # 处理0值
        normalized_blurred_grid = blurred_grid_data/blurred_grid_weight
        mask = blurred_grid_weight < -1
        normalized_blurred_grid[mask] = 0.  # 赋0

        # upsample
        grid_xx_r = (xx / sampling_spatial) + padding_xy + 1.
        grid_yy_r = (yy / sampling_spatial) + padding_xy + 1.
        grid_zz_r = (edge - edge_min) / sampling_range + padding_z + 1.

        n_x, n_y, n_z = normalized_blurred_grid.shape  # 网格中有多少离散采样点
        points = (np.arange(n_x), np.arange(n_y), np.arange(n_z))

        x_i = (grid_xx_r, grid_yy_r, grid_zz_r)

        output = interpolate.interpn(points, 
                                    normalized_blurred_grid, 
                                    x_i, 
                                    method="linear")
        return output

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
        


    