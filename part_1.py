import cv2
import os.path as osp
import numpy as np

from image_processing_lib.cli_image_argument import get_image_path
from image_processing_lib.windows_manager import create_two_windows
from image_processing_lib.time_comparing import get_time
from part_1_lib.algCanny import canny

if __name__ == "__main__":
    image_path = get_image_path(
        default_path=osp.join(
            osp.dirname(__file__),
            "part_1_lib/src/google.jpg",
        )
    )

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
result_img = canny(img)
create_two_windows(np.uint8(result_img), np.uint8(img), 'canny', 'source image')
cv2.waitKey()
