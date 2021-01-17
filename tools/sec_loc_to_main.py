import os
import re
import numpy as np

path = "G:\\gazenet-master\\test_data_sec_1"
point_num = 9


def _extract_target_from_gazefilename(imagepath):
    filename_regex = '(\d+)_(\d.\d+)_(\d.\d+)'
    m = re.search(filename_regex, imagepath)
    index = int(m.group(1))
    gaze_x = float(m.group(2))
    gaze_y = float(m.group(3))
    return index, gaze_x, gaze_y


def cal_LRTB(marker_pos):
    LRTB = np.zeros([point_num, 4])
    for i in range(point_num):
        LRTB[i][0] = (marker_pos[i][0][0] + marker_pos[i][1][0]) / 2
        LRTB[i][1] = (marker_pos[i][2][0] + marker_pos[i][3][0]) / 2
        LRTB[i][2] = (marker_pos[i][1][1] + marker_pos[i][3][1]) / 2
        LRTB[i][3] = (marker_pos[i][0][1] + marker_pos[i][2][1]) / 2
    return LRTB


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for dataset in os.listdir(path):
        dataset_path = path + '\\' + dataset
        marker_pos_path = dataset_path + '\\' + "marker_pos.npy"
        sec_marker_pos_path = dataset_path + '\\' + "sec_marker_pos.npy"
        marker_pos = np.load(marker_pos_path)
        sec_marker_pos = np.load(sec_marker_pos_path)
        marker_LRTB = cal_LRTB(marker_pos)
        sec_marker_LRTB = cal_LRTB(sec_marker_pos)

        for point in os.listdir(dataset_path):
            if not os.path.isdir(dataset_path + '\\' + point):
                continue
            else:
                i, gaze_x, gaze_y = _extract_target_from_gazefilename(point)
                i = i - 1
                point_path = dataset_path + '\\' + point
                x = marker_LRTB[i][0] + (gaze_x - sec_marker_LRTB[i][0]) / abs(
                    sec_marker_LRTB[i][0] - sec_marker_LRTB[i][1]) * abs(marker_LRTB[i][0] - marker_LRTB[i][1])
                y = marker_LRTB[i][3] + (gaze_y - sec_marker_LRTB[i][3]) / abs(
                    sec_marker_LRTB[i][2] - sec_marker_LRTB[i][3]) * abs(marker_LRTB[i][2] - marker_LRTB[i][3])
                new_name = ("%d_%f_%f" % (i+1, x, y))
                new_path = dataset_path + '\\' + new_name
                os.rename(point_path, new_path)
                print(point + '  -->  ' + new_name)
        print('ok')


