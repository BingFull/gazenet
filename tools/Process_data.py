import csv
import numpy
import math

imgWidth = 1280
imgHight = 720
distance = 45
confidence_threshold = 0.7
point_num = 9
add_point_num = 3
bt_threshold_w = 0.6
lr_threshold_w = 0.5
bt_threshold_m = 0.6
lr_threshold_m = 0.5
gaze_position_path = "gaze_positions.csv"
sec_gaze_position_path = "sec_gaze_positions.csv"
marker_pos_path = "markers_pos.csv"
sec_marker_pos_path = "sec_markers_pos.csv"
accuracy_csv_path = "result.csv"

def get_start_end_frame():
    frame_index = numpy.zeros([point_num + add_point_num, 2])
    for i in range(point_num + add_point_num):
        start, end = input("input start & end frame %d: " % i).split()
        frame_index[i][0] = start
        frame_index[i][1] = end
    return frame_index


def get_len_pixel(cam):
    len_pixel = numpy.zeros(2)
    if cam == "W_cam":
        len_pixel = input("please input Len/Pixel:(W_cam_x W_cam_y) ").split()
    elif cam == "M_cam":
        len_pixel = input("please input sec_Len/Pixel:(M_cam_x M_cam_y) ").split()
    len_pixel[0] = float(len_pixel[0])
    len_pixel[1] = float(len_pixel[1])
    return len_pixel

def get_groundtruth(cam):
    if cam == "W_cam":
        Groundtruth = numpy.zeros([point_num + add_point_num, 2])
        for i in range(point_num + add_point_num):
            Groundtruth[i] = input("please input W_cam groundtruth:(X_pixel, Y_pixel) ").split()
            Groundtruth[i][0] = Groundtruth[i][0] / imgWidth
            Groundtruth[i][1] = Groundtruth[i][1] / imgHight

    if cam == "M_cam":
        Groundtruth = numpy.zeros([point_num, 2])
        for i in range(point_num):
            Groundtruth[i] = input("please input M_cam groundtruth:(X_pixel, Y_pixel) ").split()
            Groundtruth[i][0] = Groundtruth[i][0] / imgWidth
            Groundtruth[i][1] = Groundtruth[i][1] / imgHight

    return Groundtruth


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

    average_match = numpy.mean(match_accuracy, axis=0)
    stdev_match = numpy.std(match_accuracy, axis=0)
    average_w = numpy.mean(accuracy_w, axis=0)
    stdev_w = numpy.std(accuracy_w, axis=0)
    average_m = numpy.mean(accuracy_m, axis=0)
    stdev_m = numpy.std(accuracy_m, axis=0)
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

    if add_point_num != 0:
        for index in range(add_point_num):
            dict_writer.writerow(
                {
                    "User": index + point_num + 1,
                    "match_H": " ",
                    "match_V": " ",
                    "match_N": " ",
                    "w_cam_H": accuracy_w[index + point_num][0],
                    "w_cam_V": accuracy_w[index + point_num][1],
                    "w_cam_N": accuracy_w[index + point_num][2],
                    "m_cam_H": " ",
                    "m_cam_V": " ",
                    "m_cam_N": " "
                }
            )

        average_w_add = numpy.mean(accuracy_w[point_num:], axis=0)
        stdev_w_add = numpy.std(accuracy_w[point_num:], axis=0)

        dict_writer.writerow(
            {
                "User": "Average_add",
                "match_H": " ",
                "match_V": " ",
                "match_N": " ",
                "w_cam_H": average_w_add[0],
                "w_cam_V": average_w_add[1],
                "w_cam_N": average_w_add[2],
                "m_cam_H": " ",
                "m_cam_V": " ",
                "m_cam_N": " "
            }
        )

        dict_writer.writerow(
            {
                "User": "Stdev_add",
                "match_H": " ",
                "match_V": " ",
                "match_N": " ",
                "w_cam_H": stdev_w_add[0],
                "w_cam_V": stdev_w_add[1],
                "w_cam_N": stdev_w_add[2],
                "m_cam_H": " ",
                "m_cam_V": " ",
                "m_cam_N": " "
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


def read_gaze_data(gaze_position_path, frame_index):
    gaze_pos = [[] for i in range(point_num + add_point_num)]
    result = []
    record_timestamp = numpy.zeros([point_num + add_point_num, 2])
    csvfile = open(gaze_position_path, "r")
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for row in rows:
        if row[0] == 'world_timestamp':
            continue
        if float(row[2]) >= confidence_threshold:
            for i in range(point_num + add_point_num):
                if float(row[1]) >= frame_index[i][0] and float(row[1]) <= frame_index[i][1]:
                    if record_timestamp[i][0] == 0:
                        record_timestamp[i][0] = row[0]
                    gaze_pos[i].append([row[3], row[4]])
                    record_timestamp[i][1] = row[0]
                    break
    csvfile.close()
    for i in range(point_num + add_point_num):
        matirx = numpy.array(gaze_pos[i], dtype='float_')
        average_x, average_y = numpy.mean(matirx, axis=0)
        stdev_x, stdev_y = numpy.std(matirx, axis=0)
        count = len(gaze_pos[i])
        result.append([average_x, stdev_x, average_y, stdev_y, count, float(record_timestamp[i][0]), float(record_timestamp[i][1])])
    return result


def read_sec_gaze_data(sec_gaze_position_path, gaze_data_result):
    gaze_pos = [[] for i in range(point_num)]
    result = []
    record_timestamp = numpy.zeros([point_num, 2])
    csvfile = open(sec_gaze_position_path, "r")
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for row in rows:
        if row[0] == 'world_timestamp':
            continue
        if float(row[2]) >= confidence_threshold:
            for i in range(point_num):
                if float(row[0]) >= gaze_data_result[i][5] and float(row[0]) <= gaze_data_result[i][6]:
                    if record_timestamp[i][0] == 0:
                        record_timestamp[i][0] = row[0]
                    gaze_pos[i].append([row[3], row[4]])
                    record_timestamp[i][1] = row[0]
                    break
    csvfile.close()
    for i in range(point_num):
        matirx = numpy.array(gaze_pos[i], dtype='float_')
        average_x, average_y = numpy.mean(matirx, axis=0)
        stdev_x, stdev_y = numpy.std(matirx, axis=0)
        count = len(gaze_pos[i])
        result.append([average_x, stdev_x, average_y, stdev_y, count, float(record_timestamp[i][0]),
                       float(record_timestamp[i][1])])
    return result


def read_marker_pos(marker_pos_path, gaze_data_result, cam):
    result = numpy.zeros([point_num, 4, 2])
    count = numpy.zeros([point_num, 4])
    csvfile = open(marker_pos_path, "r")
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    all_markers = []

    if cam == "W_cam":
        bt_threshold = bt_threshold_w
        lr_threshold = lr_threshold_w
    elif cam == "M_cam":
        bt_threshold = bt_threshold_m
        lr_threshold = lr_threshold_m

    for row in rows:
        if row[0] == 'timestamp':
            continue
        result.append([float(row[1]), float(row[2])])
        if row[3] == '1':
            for i in range(point_num):
                if float(row[0]) >= gaze_data_result[i][5] and float(row[0]) <= gaze_data_result[i][6]:
                    if float(row[1]) <= lr_threshold and float(row[2]) <= bt_threshold:
                        result[i][0][0] += float(row[1])
                        result[i][0][1] += float(row[2])
                        count[i][0] += 1
                        break
                    elif float(row[1]) <= lr_threshold and float(row[2]) >= bt_threshold:    # result[i][0][0]: LB_X
                        result[i][1][0] += float(row[1])                   # result[i][0][1]: LB_Y
                        result[i][1][1] += float(row[2])                   # result[i][1][0]: LT_X
                        count[i][1] += 1                                   # result[i][1][1]: LT_Y
                        break                                              # result[i][2][0]: RB_X
                    elif float(row[1]) >= lr_threshold and float(row[2]) <= bt_threshold:    # result[i][2][1]: RB_Y
                        result[i][2][0] += float(row[1])                   # result[i][3][0]: RT_X
                        result[i][2][1] += float(row[2])                   # result[i][3][1]: RT_Y
                        count[i][2] += 1
                        break
                    else:
                        result[i][3][0] += float(row[1])
                        result[i][3][1] += float(row[2])
                        count[i][3] += 1
                        break
    csvfile.close()
    for i in range(point_num):
        for j in range(4):
            result[i][j][0] = result[i][j][0] / count[i][j]
            result[i][j][1] = result[i][j][1] / count[i][j]
    return result


def cal_LRTB(marker_pos):
    LRTB = numpy.zeros([point_num, 4])
    for i in range(point_num):
        LRTB[i][0] = (marker_pos[i][0][0] + marker_pos[i][1][0]) / 2
        LRTB[i][1] = (marker_pos[i][2][0] + marker_pos[i][3][0]) / 2
        LRTB[i][2] = (marker_pos[i][1][1] + marker_pos[i][3][1]) / 2
        LRTB[i][3] = (marker_pos[i][0][1] + marker_pos[i][2][1]) / 2
    return LRTB


def process_data(len_pixel, sec_len_pixel, w_cam_gd, m_cam_gd, gaze_data_result, sec_gaze_data_result, marker_LRTB, sec_marker_LRTB):
    match_gaze_pos = numpy.zeros([point_num, 2])
    accuracy_w = numpy.zeros([point_num + add_point_num, 3])
    accuracy_m = numpy.zeros([point_num, 3])
    match_accuracy = numpy.zeros([point_num, 3])
    for i in range(point_num):
        match_gaze_pos[i][0] = marker_LRTB[i][0] + (sec_gaze_data_result[i][0] - sec_marker_LRTB[i][0]) / abs(sec_marker_LRTB[i][0] - sec_marker_LRTB[i][1]) * abs(marker_LRTB[i][0] - marker_LRTB[i][1])
        match_gaze_pos[i][1] = marker_LRTB[i][3] + (sec_gaze_data_result[i][2] - sec_marker_LRTB[i][3]) / abs(sec_marker_LRTB[i][2] - sec_marker_LRTB[i][3]) * abs(marker_LRTB[i][2] - marker_LRTB[i][3])
        accuracy_w[i] = cal_accuracy(i, w_cam_gd, gaze_data_result, len_pixel, distance, type="no match")
        accuracy_m[i] = cal_accuracy(i, m_cam_gd, sec_gaze_data_result, sec_len_pixel, distance, type="no match")
        match_accuracy[i] = cal_accuracy(i, w_cam_gd, match_gaze_pos, len_pixel, distance, type="match")

    for i in range(add_point_num):
        accuracy_w[i + point_num] = cal_accuracy(i + point_num, w_cam_gd, gaze_data_result, len_pixel, distance, type="no match")
    write_accuracy_csv(accuracy_w, accuracy_m, match_accuracy, accuracy_csv_path)


def cal_accuracy(index, groundtruth, gaze_pos, len_pixel, dis, type):
    result = numpy.zeros(3)
    if type == "no match":
        result[0] = math.atan(
            abs(gaze_pos[index][0] - groundtruth[index][0]) * imgWidth * len_pixel[0] / dis
        ) * 180 / math.pi
        result[1] = math.atan(
            abs(gaze_pos[index][2] - groundtruth[index][1]) * imgHight * len_pixel[1] / dis
        ) * 180 / math.pi
        result[2] = math.sqrt(pow(result[0], 2) + pow(result[1], 2))
    elif type == "match":
        result[0] = math.atan(
            abs(gaze_pos[index][0] - groundtruth[index][0]) * imgWidth * len_pixel[0] / dis
        ) * 180 / math.pi
        result[1] = math.atan(
            abs(gaze_pos[index][1] - groundtruth[index][1]) * imgHight * len_pixel[1] / dis
        ) * 180 / math.pi
        result[2] = math.sqrt(pow(result[0], 2) + pow(result[1], 2))
    return result


if __name__ == "__main__":
    len_pixel = get_len_pixel(cam="W_cam")
    sec_len_pixel = get_len_pixel(cam="M_cam")
    W_cam_GD = get_groundtruth(cam="W_cam")
    M_cam_GD = get_groundtruth(cam="M_cam")
    frame_index = get_start_end_frame()
    gaze_data_result = read_gaze_data(gaze_position_path, frame_index)
    sec_gaze_result = read_sec_gaze_data(sec_gaze_position_path, gaze_data_result)
    marker_pos = read_marker_pos(marker_pos_path, gaze_data_result, cam="W_cam")
    sec_marker_pos = read_marker_pos(sec_marker_pos_path, gaze_data_result, cam="M_cam")

    # sec_marker_pos[3][0] = [302 / 1280, 36 / 720]
    # sec_marker_pos[4][0] = [310 / 1280, 37 / 720]
    # sec_marker_pos[6][0] = [273 / 1280, 35 / 720]
    # sec_marker_pos[7][0] = [263 / 1280, 39 / 720]
    # sec_marker_pos[8][0] = [254 / 1280, 42 / 720]


    # sec_marker_pos[3][0] = [318 / 1280, 24 / 720]
    # sec_marker_pos[3][3] = sec_marker_pos[3][2]
    # sec_marker_pos[3][2] = [1035 / 1280, 30 / 720]
    # sec_marker_pos[4][0] = [319 / 1280, 27 / 720]
    # sec_marker_pos[4][3] = sec_marker_pos[4][2]
    # sec_marker_pos[4][2] = [1036 / 1280, 31 / 720]
    # sec_marker_pos[5][0] = [326 / 1280, 21 / 720]
    # sec_marker_pos[5][3] = sec_marker_pos[5][2]
    # sec_marker_pos[5][2] = [1043 / 1280, 26 / 720]
    # sec_marker_pos[6][0] = [328 / 1280, 17 / 720]
    # sec_marker_pos[6][3] = sec_marker_pos[6][2]
    # sec_marker_pos[6][2] = [1044 / 1280, 18 / 720]
    # sec_marker_pos[7][0] = [327 / 1280, 10 / 720]
    # sec_marker_pos[7][3] = sec_marker_pos[7][2]
    # sec_marker_pos[7][2] = [1042 / 1280, 14 / 720]
    # sec_marker_pos[8][0] = [310 / 1280, 8 / 720]
    # sec_marker_pos[8][3] = sec_marker_pos[8][2]
    # sec_marker_pos[8][2] = [1026 / 1280, 16 / 720]

    marker_LRTB = cal_LRTB(marker_pos)
    sec_marker_LRTB = cal_LRTB(sec_marker_pos)

    process_data(len_pixel, sec_len_pixel, W_cam_GD, M_cam_GD, gaze_data_result, sec_gaze_result, marker_LRTB, sec_marker_LRTB)

