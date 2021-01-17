import re
import os
import numpy as np
import csv
import math

point_num = 9
imgWidth = 1280
imgHight = 720
distance = 45
user = "User35"
accuracy_csv_path = "result_" + user + ".csv"


def _extract_target_from_gazefilename(dirpath, filename_regex='(\d.\d+)_(\d.\d+)_(\d.\d+)'):
    filename_regex = '(\d+)_(\d.\d+)_(\d.\d+)'
    # foldername = imagepath.parent.parent.name
    m = re.search(filename_regex, dirpath)
    index = int(m.group(1))
    gaze_x = float(m.group(2))
    gaze_y = float(m.group(3))
    return index, gaze_x, gaze_y


def get_groundtruth(path, cam):
    if cam == "W_cam":
        Groundtruth = np.zeros([point_num + 3, 2])
    elif cam == "M_cam":
        Groundtruth = np.zeros([point_num, 2])
    for dir in os.listdir(path):
        if(not os.path.isdir(path + '\\' + dir)):
            continue
        index, gaze_x, gaze_y = _extract_target_from_gazefilename(dir)
        Groundtruth[index-1][0] = gaze_x
        Groundtruth[index-1][1] = gaze_y
    return Groundtruth


def read_gaze_data(path, cam):
    datanames = os.listdir(path)
    if cam == "W_cam":
        gaze_pos = [[] for i in range(point_num + 3)]
    elif cam == "M_cam":
        gaze_pos = [[] for i in range(point_num)]

    for dataname in datanames:
        if os.path.splitext(dataname)[1] == '.csv':  # 目录下包含.json的文件
            name = os.path.splitext(dataname)[1]
            index = int(dataname.split('_')[0])
            result = []
            csvfile = open(path + '\\' + dataname, "r")
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            for row in rows:
                if row[0] == 'Groundtruth_x':
                    continue
                gaze_pos[index-1].append([row[2], row[3]])
            csvfile.close()
    for i in range(point_num):
        matirx = np.array(gaze_pos[i], dtype='float_')
        average_x, average_y = np.mean(matirx, axis=0)
        stdev_x, stdev_y = np.std(matirx, axis=0)
        count = len(gaze_pos[i])
        result.append([average_x, stdev_x, average_y, stdev_y, count])
    return result


def cal_LRTB(marker_pos):
    LRTB = np.zeros([point_num, 4])
    for i in range(point_num):
        LRTB[i][0] = (marker_pos[i][0][0] + marker_pos[i][1][0]) / 2
        LRTB[i][1] = (marker_pos[i][2][0] + marker_pos[i][3][0]) / 2
        LRTB[i][2] = (marker_pos[i][1][1] + marker_pos[i][3][1]) / 2
        LRTB[i][3] = (marker_pos[i][0][1] + marker_pos[i][2][1]) / 2
    return LRTB


def write_accuracy_csv(accuracy_w, accuracy_m, match_accuracy, accuracy_csv_path):
    fileheader = [
        "User", "match_H", "match_V", "match_N", "w_cam_H", "w_cam_V", "w_cam_N", "m_cam_H", "m_cam_V", "m_cam_N"
    ]
    csvfile = open(accuracy_csv_path, "w", newline='')
    dict_writer = csv.DictWriter(csvfile, fileheader)
    dict_writer.writeheader()
    for index in range(point_num):
        dict_writer.writerow(
            {
                "User": index+1,
                "match_H": match_accuracy[index][0],
                "match_V": match_accuracy[index][1],
                "match_N": match_accuracy[index][2],
                "w_cam_H": accuracy_w[index][0],
                "w_cam_V": accuracy_w[index][1],
                "w_cam_N": accuracy_w[index][2],
                "m_cam_H": accuracy_m[index][0],
                "m_cam_V": accuracy_m[index][1],
                "m_cam_N": accuracy_m[index][2]
            }
        )

    average_match = np.mean(match_accuracy, axis=0)
    stdev_match = np.std(match_accuracy, axis=0)
    average_w = np.mean(accuracy_w, axis=0)
    stdev_w = np.std(accuracy_w, axis=0)
    average_m = np.mean(accuracy_m, axis=0)
    stdev_m = np.std(accuracy_m, axis=0)
    dict_writer.writerow(
        {
            "User": "Average",
            "match_H": average_match[0],
            "match_V": average_match[1],
            "match_N": average_match[2],
            "w_cam_H": average_w[0],
            "w_cam_V": average_w[1],
            "w_cam_N": average_w[2],
            "m_cam_H": average_m[0],
            "m_cam_V": average_m[1],
            "m_cam_N": average_m[2]
        }
    )

    dict_writer.writerow(
        {
            "User": "Stdev",
            "match_H": stdev_match[0],
            "match_V": stdev_match[1],
            "match_N": stdev_match[2],
            "w_cam_H": stdev_w[0],
            "w_cam_V": stdev_w[1],
            "w_cam_N": stdev_w[2],
            "m_cam_H": stdev_m[0],
            "m_cam_V": stdev_m[1],
            "m_cam_N": stdev_m[2]
        }
    )

    dict_writer.writerow(
        {
            "User": " ",
            "match_H": " ",
            "match_V": " ",
            "match_N": " ",
            "w_cam_H": " ",
            "w_cam_V": " ",
            "w_cam_N": " ",
            "m_cam_H": " ",
            "m_cam_V": " ",
            "m_cam_N": " "
        }
    )
    csvfile.close()


def process_data(len_pixel, sec_len_pixel, w_cam_gd, m_cam_gd, gaze_data_result, sec_gaze_data_result, marker_LRTB, sec_marker_LRTB):
    match_gaze_pos = np.zeros([point_num, 2])
    accuracy_w = np.zeros([point_num, 3])
    accuracy_m = np.zeros([point_num, 3])
    match_accuracy = np.zeros([point_num, 3])
    for i in range(point_num):
        match_gaze_pos[i][0] = marker_LRTB[i][0] + (sec_gaze_data_result[i][0] - sec_marker_LRTB[i][0]) / abs(sec_marker_LRTB[i][0] - sec_marker_LRTB[i][1]) * abs(marker_LRTB[i][0] - marker_LRTB[i][1])
        match_gaze_pos[i][1] = marker_LRTB[i][3] + (sec_gaze_data_result[i][2] - sec_marker_LRTB[i][3]) / abs(sec_marker_LRTB[i][2] - sec_marker_LRTB[i][3]) * abs(marker_LRTB[i][2] - marker_LRTB[i][3])
        accuracy_w[i] = cal_accuracy(i, w_cam_gd, gaze_data_result, len_pixel, distance, type="no match")
        accuracy_m[i] = cal_accuracy(i, m_cam_gd, sec_gaze_data_result, sec_len_pixel, distance, type="no match")
        match_accuracy[i] = cal_accuracy(i, w_cam_gd, match_gaze_pos, len_pixel, distance, type="match")
    write_accuracy_csv(accuracy_w, accuracy_m, match_accuracy, accuracy_csv_path)


def cal_accuracy(index, groundtruth, gaze_pos, len_pixel, dis, type):
    result = np.zeros(3)
    if type == "no match":
        result[0] = math.atan(
            abs(gaze_pos[index][0] - groundtruth[index][0]) * imgWidth * len_pixel[0] / dis
        ) * 180 / math.pi
        result[1] = math.atan(
            abs(gaze_pos[index][2] - groundtruth[index][1]) * imgHight * len_pixel[1] / dis
        ) * 180 / math.pi
        result[2] = math.sqrt(pow(result[0], 2) + pow(result[1], 2))
    elif type == "match":
        a = gaze_pos[index][0]
        b = groundtruth[index][0]
        c = imgWidth
        d = len_pixel[0]
        result[0] = math.atan(
            abs(gaze_pos[index][0] - groundtruth[index][0]) * imgWidth * len_pixel[0] / dis
        ) * 180 / math.pi
        result[1] = math.atan(
            abs(gaze_pos[index][1] - groundtruth[index][1]) * imgHight * len_pixel[1] / dis
        ) * 180 / math.pi
        result[2] = math.sqrt(pow(result[0], 2) + pow(result[1], 2))
    return result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    len_pixel = [0.0525, 0.05928]
    sec_len_pixel = [0.02564, 0.02564]
    path = "G:\\gazenet-master\\test_data\\" + user
    # path = "G:\\experimental_records\\08_29\\" + user
    path_sec = "G:\\gazenet-master\\test_data_for_sec\\" + user
    marker_path = "G:\\gazenet-master\\test_data\\" + user + "\\marker_pos.npy"
    # marker_path = "G:\\experimental_records\\08_29\\" + user + "\\marker_pos.npy"

    sec_marker_path = "G:\\gazenet-master\\test_data_for_sec\\" + user + "\\sec_marker_pos.npy"
    W_cam_GD = get_groundtruth(path, cam='W_cam')
    M_cam_GD = get_groundtruth(path_sec, cam='M_cam')
    gaze_data_result = read_gaze_data(path, cam='W_cam')
    sec_gaze_result = read_gaze_data(path_sec, cam='M_cam')
    marker_pos = np.load(marker_path)
    sec_marker_pos = np.load(sec_marker_path)
    marker_LRTB = cal_LRTB(marker_pos)
    sec_marker_LRTB = cal_LRTB(sec_marker_pos)
    process_data(len_pixel, sec_len_pixel, W_cam_GD, M_cam_GD, gaze_data_result, sec_gaze_result, marker_LRTB,
                 sec_marker_LRTB)
    print('Done!')




