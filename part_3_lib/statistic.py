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

    intersect = cv2.threshold(
                    actual_img + ref_img,
                    0,
                    255,
                    cv2.THRESH_BINARY)[1]
    union = cv2.threshold(
                    actual_img * ref_img,
                    0,
                    255,
                    cv2.THRESH_BINARY)[1]

    create_two_windows(
        intersect,
        union,
        'intersect',
        'union',
    )

    iou = (np.count_nonzero(intersect == 0) /
           np.count_nonzero(union == 0))
    print("Intersection over union: ", iou)
    
    fp = (np.count_nonzero((actual_img == 0) & (actual_img != ref_img)) /
           np.count_nonzero(union == 0))
    print("False positive: ", fp)
    
    fn = (np.count_nonzero((ref_img == 0) & (actual_img != ref_img)) /
           np.count_nonzero(union == 0))
    print("False negative: ", fn)
    
    return iou, fp, fn
