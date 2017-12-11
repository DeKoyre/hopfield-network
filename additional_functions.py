import cv2
import numpy as np
from os.path import isfile


def image_to_array(path,
                   background,
                   range_param,
                   target_is_darker):
    if isfile(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        background_int = []
        for i in range(0, len(background)):
            background_int.append(int(background[i], 16))
        if background_int[0] == background_int[1] & background_int[0] == background_int[2]:
            background_int = background_int[0]
        else:
            background_int = int(round((background_int[0] + background_int[1] + background_int[2]) / 3))
        if target_is_darker:
            threshold_param = background_int - range_param
        else:
            threshold_param = background_int + range_param
        shape = gray.shape
        target = gray.copy()
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                if target_is_darker:
                    if (target[i, j]) > threshold_param:
                        target[i, j] = 0
                    else:
                        target[i, j] = 255
                else:
                    if (target[i, j]) <= threshold_param:
                        target[i, j] = 0
                    else:
                        target[i, j] = 255
        target_bp = target.copy().astype(int)
        np.place(target_bp, target_bp == 0, -1)
        np.place(target_bp, target_bp == 255, 1)

        return target_bp.flatten(order='C')[np.newaxis]
        # return target_bp.flatten(order='C')[np.newaxis]
    else:
        print 'Failed to load file at: '
        print path
