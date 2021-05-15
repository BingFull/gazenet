import csv
import numpy
import math
from scipy.cluster.vq import kmeans
from collections import Counter
import cv2
import os

imgWidth = 1280
imgHight = 720
confidence_threshold = 0.7
point_num = 9
add_point_num = 3
bt_threshold_w = 0.6
lr_threshold_w = 0.5
bt_threshold_m = 0.5
lr_threshold_m = 0.5
user = "User26"
gaze_position_path = "data_n" + "/" + user + "/" + "gaze_positions.csv"
pupil_position_path = "data_n" + "/" + user + "/" + "pupil_positions.csv"
marker_pos_path = "data_n" + "/" + user + "/" + "markers_pos.csv"
parameter_path = "data_n" + "/" + user + "/" + "parameter.csv"
eye_timestamps = "data_n" + "/" + user + "/" + "eye1_timestamps.npy"
video_path = "data_n" + "/" + user + "/" + "eye1.mp4"
rst_path = 'frames_new' + "/" + user
marker_pos_save_path = "data_n" + "/" + user + "/" + "marker_pos.npy"


def read_parameter():
    parameters = []
    csvfile = open(parameter_path, "r")
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for row in rows:
        parameters.append([float(row[0]), float(row[1])])
    csvfile.close()
    return parameters


def get_start_end_frame(parameters):
    frame_index = []
    for i in range(2, point_num + add_point_num + 2):
        frame_index.append([parameters[i][0], parameters[i][1]])
    return frame_index


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
        result.append([average_x, stdev_x, average_y, stdev_y, count, float(record_timestamp[i][0]),
                       float(record_timestamp[i][1])])
    return result


def read_pupil_data(pupil_position_path, frame_index, eye_id):
    pupil_pos = [[] for i in range(point_num + add_point_num)]
    result = []
    record_timestamp = numpy.zeros([point_num + add_point_num, 2])
    csvfile = open(pupil_position_path, "r")
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for row in rows:
        if row[0] == 'world_timestamp':
            continue
        if float(row[3]) >= confidence_threshold and int(row[2]) == eye_id:
            for i in range(point_num + add_point_num):
                if float(row[1]) >= frame_index[i][0] and float(row[1]) <= frame_index[i][1]:
                    if record_timestamp[i][0] == 0:
                        record_timestamp[i][0] = row[0]
                    pupil_pos[i].append([row[4], row[5]])
                    record_timestamp[i][1] = row[0]
                    break
    csvfile.close()
    for i in range(point_num + add_point_num):
        matirx = numpy.array(pupil_pos[i], dtype='float_')
        average_x, average_y = numpy.mean(matirx, axis=0)
        stdev_x, stdev_y = numpy.std(matirx, axis=0)
        count = len(pupil_pos[i])
        result.append([average_x, stdev_x, average_y, stdev_y, count, float(record_timestamp[i][0]),
                       float(record_timestamp[i][1])])
    return result


def read_marker_pos_new(marker_pos_path, gaze_data_result, cam):
    result_devide = numpy.zeros([point_num + add_point_num, 4, 2])
    count = numpy.zeros([point_num + add_point_num, 4])
    csvfile = open(marker_pos_path, "r")
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    if cam == "W_cam":
        bt_threshold = bt_threshold_w
        lr_threshold = lr_threshold_w
        markers_num = point_num + add_point_num
    elif cam == "M_cam":
        bt_threshold = bt_threshold_m
        lr_threshold = lr_threshold_m
        markers_num = point_num

    markers = []
    for i in range(markers_num):
        markers.append([])

    tmp_time = 0
    tmp_count = -1
    tmp_i = 0
    for row in rows:
        if row[0] == 'timestamp':
            continue
        for i in range(markers_num):
            if float(row[0]) >= gaze_data_result[i][5] and float(row[0]) <= gaze_data_result[i][6]:
                if tmp_i != i:
                    tmp_count = -1
                    tmp_i = i
                if float(row[0]) != tmp_time:
                    tmp_count += 1
                    markers[i].append([])
                    markers[i][tmp_count].append([float(row[0]), float(row[1]), float(row[2])])
                    tmp_time = float(row[0])
                else:
                    markers[i][tmp_count].append([float(row[0]), float(row[1]), float(row[2])])
    csvfile.close()
    return markers


def read_marker_pos_save(marker_pos_path, gaze_data_result):
    result = numpy.zeros([point_num, 4, 2])
    count = numpy.zeros([point_num, 4])
    csvfile = open(marker_pos_path, "r")
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for row in rows:
        if row[3] == '1':
            for i in range(point_num):
                if float(row[0]) >= gaze_data_result[i][5] and float(row[0]) <= gaze_data_result[i][6]:
                    if float(row[1]) <= 0.5 and float(row[2]) <= 0.6:
                        result[i][0][0] += float(row[1])
                        result[i][0][1] += float(row[2])
                        count[i][0] += 1
                        break
                    elif float(row[1]) <= 0.5 and float(row[2]) >= 0.6:    # result[i][0][0]: LB_X
                        result[i][1][0] += float(row[1])                   # result[i][0][1]: LB_Y
                        result[i][1][1] += float(row[2])                   # result[i][1][0]: LT_X
                        count[i][1] += 1                                   # result[i][1][1]: LT_Y
                        break                                              # result[i][2][0]: RB_X
                    elif float(row[1]) >= 0.5 and float(row[2]) <= 0.6:    # result[i][2][1]: RB_Y
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


def read_marker_pos(marker_pos_path, gaze_data_result, cam):
    csvfile = open(marker_pos_path, "r")
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    if cam == "W_cam":
        markers_num = point_num + add_point_num
    elif cam == "M_cam":
        markers_num = point_num

    markers = []
    for i in range(markers_num):
        markers.append([])

    for row in rows:
        if row[0] == 'timestamp':
            continue
        for i in range(markers_num):
            if float(row[0]) >= gaze_data_result[i][5] and float(row[0]) <= gaze_data_result[i][6]:
                markers[i].append([float(row[1]), float(row[2])])
    return markers


def sort_markers(centroid, cam):
    if cam == "W_cam":
        markers_num = point_num + add_point_num
    elif cam == "M_cam":
        markers_num = point_num

    markers_list_sort_y = sorted(centroid, key=lambda x: x[1], reverse=True)
    markers_list_sort_y = numpy.array(markers_list_sort_y)
    markers_list_sorted = []
    for i in range(0, markers_num, 3):
        markers_high = markers_list_sort_y[i:i + 3]
        markers_high = numpy.array(markers_high)
        if i % 2 == 0:
            markers_high_sorted = sorted(markers_high, key=lambda x: x[0])
        else:
            markers_high_sorted = sorted(markers_high, key=lambda x: x[0], reverse=True)
        markers_high_sorted = numpy.array(markers_high_sorted)
        for j in range(3):
            markers_list_sorted.append(markers_high_sorted[j])
    return markers_list_sorted


def video_to_img(videopath, rst_path, eye_start, eye_end, cam_GD):
    """ 将视频转换成图片 path: 视频路径 """
    cap = cv2.VideoCapture(videopath)
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        params = [2]
        if suc:
            for i in range(len(cam_GD)):
                if frame_count >= eye_start[i] + 1 and frame_count <= eye_end[i]+1:
                    tmp_loc = cam_GD[i]
                    save_path = "{}_{}_{}/".format(str(i+1), str(tmp_loc[0]), str(tmp_loc[1]))
                    save_path = rst_path + '/' + save_path
                    if not os.path.exists(save_path):  # 如果路径不存在
                        os.makedirs(save_path)
                    cv2.imwrite((save_path + '/%d.jpg') % frame_count, frame, params)
    cap.release()
    print('unlock movie: ', frame_count)
    return


def get_eye_start_end(eye_timestamps, markers):
    eye_timestamps = numpy.load(eye_timestamps)
    tmp_start = [0] * len(markers)
    tmp_end = [0] * len(markers)
    for i in range(eye_timestamps.size):
        for j in range(point_num + add_point_num):
            if eye_timestamps[i] >= markers[j][0][0][0] and eye_timestamps[i] <= markers[j][-1][0][0]:
                if i == 0:
                    tmp_start[j] = 0
                    continue
                else:
                    if i == eye_timestamps.size-1:
                        tmp_end[j] = i
                        continue
                    elif eye_timestamps[i-1] < markers[j][0][0][0]:
                        tmp_start[j] = i
                    elif eye_timestamps[i+1] > markers[j][-1][0][0]:
                        tmp_end[j] = i
    return tmp_start, tmp_end


if __name__ == "__main__":
    # aa = numpy.load(marker_pos_save_path)
    parameters = read_parameter()
    frame_index = get_start_end_frame(parameters)
    gaze_data_result = read_gaze_data(gaze_position_path, frame_index)

    # marker_pos = read_marker_pos_save(marker_pos_path, gaze_data_result)
    # numpy.save(marker_pos_save_path, marker_pos)

    markers = read_marker_pos_new(marker_pos_path, gaze_data_result, cam="W_cam")
    centroid = []
    W_cam_GD = []

    eye_start, eye_end = get_eye_start_end(eye_timestamps, markers)

    markers = read_marker_pos(marker_pos_path, gaze_data_result, cam="W_cam")

    for i in range(len(markers)):
        centroid.append([])
        centroid[i] = kmeans(numpy.array(markers[i]), point_num + add_point_num, iter=200, thresh=1e-8)[0]
        sort_m = sort_markers(centroid[i], cam="W_cam")
        W_cam_GD.append(sort_m[i])

    video_to_img(video_path, rst_path, eye_start, eye_end, W_cam_GD)
    print(1)
