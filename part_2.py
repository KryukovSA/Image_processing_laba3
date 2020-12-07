import cv2
import os.path as osp

from image_processing_lib.cli_image_argument import get_image_path
from image_processing_lib.windows_manager import create_two_windows
from image_processing_lib.time_comparing import get_time
from part_2_lib.circle_search import serch_circles, draw_circles


if __name__ == "__main__":
    image_path = get_image_path(
        default_path=osp.join(
            osp.dirname(__file__),
            "part_2_lib/src/HoughCircles.jpg",
        )
    )

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blured_img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blured_img, 50, 150, apertureSize=3)

    circles = get_time(serch_circles, edges, 10, 120, 120, 40)
    my_detected_circles = draw_circles(circles, img)

    create_two_windows(my_detected_circles, img, 'my_detected_circles', 'source image')

    circles = get_time(
        cv2.HoughCircles,
        edges,
        cv2.HOUGH_GRADIENT,
        1,
        5,
        param1=120,
        param2=40,
        minRadius=10,
        maxRadius=120,
    )
    opencv_detected_circles = draw_circles(circles[0], img)

    create_two_windows(
        opencv_detected_circles, img, 'opencv_detected_circles', 'source image'
    )
    create_two_windows(
        opencv_detected_circles,
        my_detected_circles,
        'opencv_detected_circles',
        'my_detected_circles',
    )
