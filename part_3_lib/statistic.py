import numpy as np
import cv2
import math

from part_2_lib.circle_search import draw_circles
from image_processing_lib.windows_manager import create_two_windows


def compare_circles(
    shape: tuple,
    ref_circles: list,
    actual_circles: list
):
    ref_img = np.full(shape, 10, dtype=np.uint8)
    actual_img = draw_circles(actual_circles, ref_img, thickness=-1)
    ref_img = draw_circles(ref_circles, ref_img, thickness=-1)

    intersect_img = cv2.threshold(
                    actual_img + ref_img,
                    0,
                    255,
                    cv2.THRESH_BINARY)[1]
    union_img = cv2.threshold(
                    actual_img * ref_img,
                    0,
                    255,
                    cv2.THRESH_BINARY)[1]

    iou = (np.count_nonzero(intersect_img == 0) /
           np.count_nonzero(union_img == 0))
    print("Intersection over union: ", iou)

    create_two_windows(
        intersect_img,
        union_img,
        'intersect',
        'union',
    )

    ref_img = cv2.threshold(
                    ref_img,
                    0,
                    255,
                    cv2.THRESH_BINARY)[1]
    actual_img = cv2.threshold(
                    actual_img,
                    0,
                    255,
                    cv2.THRESH_BINARY)[1]

    fp_img = np.full(shape, 255, dtype=np.uint8) - (intersect_img - actual_img)
    fn_img = np.full(shape, 255, dtype=np.uint8) - (intersect_img - ref_img)

    fp = (np.count_nonzero((actual_img == 0) & (actual_img != ref_img)) /
          np.count_nonzero(union_img == 0))
    print("False positive: ", fp)

    fn = (np.count_nonzero((ref_img == 0) & (actual_img != ref_img)) /
          np.count_nonzero(union_img == 0))
    print("False negative: ", fn)

    create_two_windows(
        fp_img,
        fn_img,
        'false positive',
        'false negative',
    )

    return iou, fp, fn


def count_circles(
    circles: list,
    threshold_same: float = 0.6
):
    diff_circles = []
    for circle in circles:
        if not same_already_in(circle, diff_circles, threshold_same):
            diff_circles.append(circle)
    return len(diff_circles)


def same_already_in(
    new_circle: tuple,
    circles: list,
    threshold_same: float
):
    have_same = False
    x0, y0, r0 = new_circle
    for circle in circles:
        x, y, r = circle
        d = math.sqrt((x0 - x)**2 + (y0 - y)**2)
        if d > r + r0:  # no intersect
            continue
        if (r0 * threshold_same > r or r0 < r * threshold_same):  # different by size
            continue
        if d < abs(r0 - r * threshold_same if r0 > r else r - r0 * threshold_same):  # same
            have_same = True
            break

    return have_same
