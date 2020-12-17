import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve


def operator_sobel(img):
    grad_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    grad_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    gorizonial = ndimage.filters.convolve(img, grad_x)
    vertical = ndimage.filters.convolve(img, grad_y)
    g_res = np.hypot(gorizonial, vertical)
    g_res = g_res / g_res.max() * 255
    theta = np.arctan2(vertical, gorizonial)
    return g_res, theta


def leave_pacification(image, teta):
    a, b = image.shape
    value = np.zeros((a, b), dtype=np.int32)
    corner = teta * 180. / 3.1415
    corner[corner < 0] += 180

    for i in range(1, a - 1):
        for j in range(1, b - 1):
            qt = 255
            rt = 255

            if (0 <= corner[i, j] < 22.5) or (157.5 <= corner[i, j] <= 180):
                qt = image[i, j + 1]
                rt = image[i, j - 1]

            elif 22.5 <= corner[i, j] < 67.5:
                qt = image[i + 1, j - 1]
                rt = image[i - 1, j + 1]

            elif 67.5 <= corner[i, j] < 112.5:
                qt = image[i + 1, j]
                rt = image[i - 1, j]

            elif 112.5 <= corner[i, j] < 157.5:
                qt = image[i - 1, j - 1]
                rt = image[i + 1, j + 1]

            if (image[i, j] >= qt) and (image[i, j] >= rt):
                value[i, j] = image[i, j]
            else:
                value[i, j] = 0

    return value


def gauss(sz, sgm=1):
    sz = int(sz) // 2
    x, y = np.mgrid[-sz:sz + 1, -sz:sz + 1]
    ideal = 1 / (2.0 * 3.1415 * sgm ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sgm ** 2))) * ideal
    return g


def thresholds(image, low_pix, heavy_pix, threshold1, threshold2):
    threshold2 = image.max() * threshold2
    threshold1 = threshold2 * threshold1
    a, b = image.shape
    result = np.zeros((a, b), dtype=np.int32)
    low = np.int32(low_pix)
    heavy = np.int32(heavy_pix)
    low_i, heavy_j = np.where(image >= threshold2)
    weak_i, weak_j = np.where((image <= threshold2) & (image >= threshold1))
    result[low_i, heavy_j] = heavy
    result[weak_i, weak_j] = low
    return result


def dependence(image, low_pix, heavy_pix):
    a, b = image.shape
    low = low_pix
    heavy = heavy_pix
    for i in range(1, a - 1):
        for j in range(1, b - 1):
            if image[i, j] == low:
                if ((image[i + 1, j - 1] == heavy) or (image[i + 1, j] == heavy) or (image[i + 1, j + 1] == heavy)
                        or (image[i, j - 1] == heavy) or (image[i, j + 1] == heavy)
                        or (image[i - 1, j - 1] == heavy) or (image[i - 1, j] == heavy) or (
                                image[i - 1, j + 1] == heavy)):
                    image[i, j] = heavy
                else:
                    image[i, j] = 0
    return image


def shades_gray(rgb):
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray_col = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    return gray_col


def algorithm_canny(image):
    image = shades_gray(image)
    image_flatten = convolve(image, gauss(5, 1))
    rug_grad, rug_teta = operator_sobel(image_flatten)
    pacification_img = leave_pacification(rug_grad, rug_teta)
    image_limit = thresholds(pacification_img, low_pix=75, heavy_pix=255, threshold1=0.05, threshold2=0.15)
    result_image = dependence(image_limit, low_pix=75, heavy_pix=255)
    return result_image
