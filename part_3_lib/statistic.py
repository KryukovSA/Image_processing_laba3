import numpy as np
import cv2

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
