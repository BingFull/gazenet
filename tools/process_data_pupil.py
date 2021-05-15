import csv
import numpy
import sys
import cv2
import os

imgWidth = 1280
imgHight = 720
confidence_threshold = 0.7
point_num = 9
add_point_num = 3

user = "User35"
gaze_position_path = "G:/gazenet_data/data_n" + "/" + user + "/" + "gaze_positions.csv"
pupil_position_path = "G:/gazenet_data/data_n" + "/" + user + "/" + "pupil_positions.csv"
marker_pos_path = "G:/gazenet_data/data_n" + "/" + user + "/" + "markers_pos.csv"
parameter_path = "G:/gazenet_data/data_n" + "/" + user + "/" + "parameter.csv"
eye_timestamps = "G:/gazenet_data/data_n" + "/" + user + "/" + "eye1_timestamps.npy"
video_path = "G:/gazenet_data/data_n" + "/" + user + "/" + "eye1.mp4"
rst_path = 'G:/gazenet_data/frames_new' + "/" + user


def read_pupil_data(pupil_position_path, eye_id):
    pupil_pos = []

    csvfile = open(pupil_position_path, "r")
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for row in rows:
        if row[0] == 'world_timestamp':
            continue
        if float(row[3]) >= confidence_threshold and int(row[2]) == eye_id:
            pupil_pos.append([row[0], row[4], row[5]])
    csvfile.close()

    return pupil_pos


def video_to_img(videopath, rst_path, eye_timestamps, pupil_pos):
    """ 将视频转换成图片 path: 视频路径 """
    cap = cv2.VideoCapture(videopath)
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    eye_timestamps = numpy.load(eye_timestamps)
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        params = [2]
        if suc:
            index = frame_count - 1
            tmp_timestamps = eye_timestamps[index]
            array_list = numpy.array(pupil_pos)
            a = array_list[:, 0]
            numbers = list(map(float, a))
            if tmp_timestamps not in numbers:
                continue
            tmp_index = numpy.where(numbers == tmp_timestamps)[0]
            save_path = "{}_{}_{}".format(str(index), pupil_pos[tmp_index[0]][1], pupil_pos[tmp_index[0]][2])
            save_path = rst_path + '/' + save_path
            if not os.path.exists(rst_path):  # 如果路径不存在
                os.makedirs(rst_path)
            cv2.imwrite(save_path + '.jpg', frame, params)
            sys.stdout.write("\r processing...: %d" % index)
            sys.stdout.flush()
    cap.release()
    print('\nunlock movie: ', frame_count)
    return


if __name__ == "__main__":

    pupil_pos = read_pupil_data(pupil_position_path, eye_id=1)

    video_to_img(video_path, rst_path, eye_timestamps, pupil_pos)
    print("Done!")
