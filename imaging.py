import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal, ndimage
import utility


class ImageInfo:
    def __init__(self, image_name="unknown", data=-1, is_show=False):
        self.name = image_name
        self.data = data
        self.size = np.shape(self.data)
        self.is_show = is_show

        if self.is_show:
            plt.imshow(self.sata)
            plt.imshow()

    def set_data(self, data):
        self.data = data
        self.size = np.shape(self.data)


def black_level_correction(image_raw, black_level, white_level, clip_range=[0, 65535]):
    '''black level correction algorithm, this function will normalize the raw image data to [0, 1] range, and clip the output to clip_range
    Args:
        image_raw: raw image data, numpy array
        black_level: black level value, int
        white_level: white level value, int
        clip_range: clip range for output image, tuple (min, max)
    '''
    image_raw, black_level, white_level = map(float, (image_raw, black_level, white_level))

    data = np.zeros(image_raw.shape, dtype=np.float32)
    data = (image_raw - black_level) / (white_level - black_level)

    data = np.clip(data, clip_range[0], clip_range[1])
    return data


def bad_pixel_correction(data, kernel_size=3, k=4, offset=0):
    '''bad pixel correction algorithm, this function will replace the bad pixel value with the median value of its neighbors
    Args:
        data: input data
        kernel_size: size of the kernel for median filter, int
    '''
    data = np.asarray(data, dtype=np.float32)  # [w, h]
    
    mean = ndimage.median_filter(data, size=kernel_size, mode='reflect')
    mad = ndimage.median_filter(np.abs(data - mean), size=kernel_size, mode='reflect')
    mad += 1e-6  # to avoid division by zero
    threshold = k * mad + offset
    bad_pixel_mask = np.abs(data - mean) > threshold

    data_pad = np.pad(data, kernel_size//2, mode='reflect')  # Pad the data to handle borders
    
    # 给原图的每个像素都找到它的8个邻居，分别是左、右、上、下、左上、右上、左下、右下
    # e.g. left[i, j] is the left neighbor of central[i, j], right[i, j] is the right neighbor of central[i, j], and so on.
    central = data_pad[1:-1, 1:-1]  # Central part of the padded data  [h, w]
    left = data_pad[1:-1, :-2]  # Left neighbors
    right = data_pad[1:-1, 2:]  # Right neighbors
    top = data_pad[:-2, 1:-1]  # Top neighbors
    bottom = data_pad[2:, 1:-1]  # Bottom neighbors
    left_top = data_pad[:-2, :-2]  # Left top neighbors
    right_top = data_pad[:-2, 2:]  # Right top neighbors
    left_bottom = data_pad[2:, :-2]  # Left bottom neighbors
    right_bottom = data_pad[2:, 2:]  # Right bottom neighbors

    pred_h = (left + right) / 2
    pred_v = (top + bottom) / 2
    pred_d1 = (left_top + right_bottom) / 2
    pred_d2 = (right_top + left_bottom) / 2

    grad_h = np.abs(left - right)  # [h, w]
    grad_v = np.abs(top - bottom)  # [h, w]
    grad_d1 = np.abs(left_top - right_bottom)  # [h, w]
    grad_d2 = np.abs(right_top - left_bottom)  # [h, w]

    grad = np.stack([grad_h, grad_v, grad_d1, grad_d2], axis=0)  # [4, h, w]
    pred = np.stack([pred_h, pred_v, pred_d1, pred_d2], axis=0)  # [4, h, w]

    min_grad_idx = np.argmin(grad, axis=0)  # [h, w]，由0，1，2，3组成，分别对应水平、垂直、左上-右下、右上-左下四个方向，是哪个说明哪个方向的梯度最小

    # data_correct = np.take_along_axis(pred, np.expand_dims(min_grad_idx, axis=0), axis=0)[0]  # [h, w]，取出对应最小梯度的预测值作为修正值
    
    data_correct = np.zeros_like(min_grad_idx, dtype=np.float32)  # [h, w]
    for i in range(min_grad_idx.shape[0]):
        for j in range(min_grad_idx.shape[1]):
            data_correct[i, j] = pred[min_grad_idx[i, j], i, j]
    data_bpc = np.copy(data)
    data_bpc[bad_pixel_mask] = data_correct[bad_pixel_mask]

    return data_bpc, bad_pixel_mask


def non_uniformity_correction(data):
    '''non-uniformity correction algorithm, this function will correct the non-uniformity of the sensor
    Args:
        data: input data
    '''
    data = np.asarray(data, dtype=np.float32)
    white_level = np.max(data)
    black_level = np.min(data)
    temp = white_level - black_level + 1e-6
    return (data - black_level) * np.average(temp) / temp


class lens_shading_correction:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    # 每个像素输出：out = gain * in + offset
    # dark_image = offset
    # flat_image = gain * in(均匀) + offset -> 算出gain
    # out_corr = (out - offset) / gain
    def flat_field_compensation(self, dark_image, flat_image):
        flat_image = np.asarray(flat_image, dtype=np.float32)
        dark_image = np.asarray(dark_image, dtype=np.float32)
        temp = flat_image - dark_image + 1e-6
        return (self.data - dark_image) * np.average(temp) / temp
    
    def approximate_methmatical_compensation(self, parameters, ):
        dark_image = np.asarray(parameters['dark_image'], dtype=np.float32)
        flat_image = np.asarray(parameters['flat_image'], dtype=np.float32)
        gain = np.average(flat_image - dark_image) / (flat_image - dark_image)
        return self.data * gain
    

class nonlinear_correction:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    # 整体调整亮度
    def lumma_adjustment(self, multiplier, clip_range=[0, 65535]):
        return np.clip(np.log10(multiplier) * self.data, clip_range[0], clip_range[1])
    
    # gamma校正，value > 1会使图像变暗，value < 1会使图像变亮
    def by_value(self, value, clip_range=[0, 65535]):
        data = np.clip(self.data, clip_range[0], clip_range[1])
        data_normalize = (data / clip_range[1]) ** value
        return np.clip(data_normalize * clip_range[1], clip_range[0], clip_range[1])

    def by_table(self, table, type="gamma", clip_range=[0, 65535]):
        gamma_table = np.loadtxt(table, dtype=np.float32)
        gamma_table = clip_range[1] * gamma_table / np.max(gamma_table)  # normalize to clip_range[1]
        linear_table = np.linspace(clip_range[0], clip_range[1], np.size(gamma_table), dtype=np.float32)

        if type == "gamma":
            return np.interp(self.data, linear_table, gamma_table)
        elif type == "degamma":
            return np.interp(self.data, gamma_table, linear_table)
        
    def by_equation(self, a, b, clip_range=[0, 65535]):
        data = np.clip(self.data, clip_range[0], clip_range[1])
        data_normalize = data / clip_range[1]
        return np.clip(clip_range[1] * (a * np.exp(b * data_normalize) + (1 - np.exp(b)) * a * data_normalize - a), clip_range[0], clip_range[1])


class tone_mapping:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    def nonlinear_masking(self, strength=1, kernel_size=5, sigma=1.0, clip_range=[0, 65535]):
        '''暗区提亮，亮区基本不变，压缩范围
        '''
        gaussian_kernel = utility.filter().gaussian(kernel_size=kernel_size, sigma=sigma)
        # 高斯模糊使得图像变得更平滑，使得后续处理颗粒感没那么强
        image_blur = signal.convolve2d(self.data, gaussian_kernel, mode='same', boundary='symm')
        image_blur = strength * image_blur / clip_range[1]
        
        '''根据image_blur的值来调整原图的亮度
        image_blur越大，说明该像素所在区域越亮，alpha越大，使得亮区不变或者亮一点点
        image_blur越小，说明该像素所在区域越暗，alpha越小，使得暗区更亮
        '''
        alpha = 0.5 ** image_blur
        return np.clip(clip_range[1] * ((self.data/clip_range[1]) ** alpha), clip_range[0], clip_range[1])
    
    def dynamic_range_compression(self, type="normal", bound=[-40., 260.], clip_range=[0, 65535]):
        '''把亮度当作是base和detail的乘积，
        压缩base，保留detail
        '''
        if type == "normal":
            edge = self.data
        elif type == "joint":
            edge = utility.edge_detection(self.data).sobel(kernel_size=3, type="gradient_magnitude")

            

if __name__ == "__main__":
    print("this is imaging module")
    a = 1
    a += 1
    print(a)