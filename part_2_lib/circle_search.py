import numpy as np
import cv2
from tqdm import tqdm

from numba import jit


def serch_circles(
    image: np.ndarray,
    min_radius: int,
    max_radius: int,
    param1: int = 150,
    param2: int = 33,
):
    radiuses = range(min_radius, max_radius)
    circles_result = []

    angle_sin_const = np.array([np.sin(angle * np.pi / 180) for angle in range(360)])
    angle_cos_const = np.array([np.cos(angle * np.pi / 180) for angle in range(360)])

    for r in tqdm(radiuses):
        cells_accumulator = get_cells_accumulator(
            r, angle_sin_const, angle_cos_const, image
        )

        cells_accumulator_max = np.amax(cells_accumulator)
        if cells_accumulator_max > param1:
            circles_result.extend(
                _scan_for_circle(cells_accumulator, image, param1, param2, r)
            )
    return circles_result


@jit
def get_cells_accumulator(r, angle_sin_const, angle_cos_const, image):
    b_const = [round(r * angle_sin_const[angle]) for angle in range(360)]
    a_const = [round(r * angle_cos_const[angle]) for angle in range(360)]
    cells_accumulator = np.zeros(image.shape, dtype=np.uint64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] == 255:
                for angle in range(360):
                    b = y - b_const[angle]
                    a = x - a_const[angle]
                    if 0 <= a < image.shape[0] and 0 <= b < image.shape[1]:
                        cells_accumulator[a][b] += 1
    return cells_accumulator


@jit
def _scan_for_circle(
    cells_accumulator: np.ndarray, image: np.ndarray, param1: int, param2: int, r: int
):
    circles_result = []
    for acc in cells_accumulator:
        if acc.all() < param1:
            acc = 0
    # cells_accumulator[cells_accumulator < param1] = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (
                cells_accumulator[i][j] >= param1
                and 0 < i < image.shape[0] - 1
                and 0 < j < image.shape[1] - 1
            ):
                mean_sum = np.mean(cells_accumulator[i - 1:i + 1, j - 1:j + 1])
                if mean_sum >= param2:
                    circles_result.append((j, i, r))
                    cells_accumulator[i:(i + 5), j:(j + 7)] = 0
    return circles_result


def draw_circles(circles: list, image: np.ndarray, thickness: int = 1):
    image_with_circles = np.copy(image)
    for vertex in circles:
        cv2.circle(image_with_circles, (vertex[0], vertex[1]), vertex[2], (0, 0, 0), thickness)
    return image_with_circles
