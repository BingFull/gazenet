import os
import shutil
import numpy as np
import re
import cv2


def _extract_target_from_gazefilename(imagepath, filename_regex='(\d.\d+)_(\d.\d+)_(\d.\d+).jpg'):
    """
    Extract the label from the image path name
    :imagepath: (Path) image path (contains target)
    :filename_regex: (string) regex used to extract gaze data from filename
    :return: tuple(int, int) gaze target
    """
    filename_regex = '(\d+)_(\d.\d+)_(\d.\d+)'
    # foldername = imagepath.parent.name
    foldername  =imagepath.split('\\')[-2]
    m = re.search(filename_regex, foldername)
    gaze_x = float(m.group(2))
    gaze_y = float(m.group(3))
    return gaze_x, gaze_y


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


def create_heatmap(data_dir_path, img_width, img_height):
    base_path = data_dir_path.replace(data_dir_path.split('\\')[-1], '')

    dir_list = os.listdir(data_dir_path)
    for dir in dir_list:
        if dir == '.DS_Store':
            os.remove(data_dir_path + '\\' + dir)
            continue
        sub_dir_list = os.listdir(data_dir_path + '\\' + dir)
        for sub_dir in sub_dir_list:
            if sub_dir == '.DS_Store':
                os.remove(data_dir_path + '\\' + dir + '\\' + sub_dir)
                continue
            file_list = os.listdir(data_dir_path + '\\' + dir + '\\' + sub_dir)
            train_dir_path = data_dir_path + '\\' + dir + '\\' + sub_dir + '\\' + 'train'
            test_dir_path = data_dir_path + '\\' + dir + '\\' + sub_dir + '\\' + 'test'

            tmp_path = base_path + 'train_data_heapmap' + '\\' + dir + '\\' + sub_dir[0]

            for file in os.listdir(train_dir_path):
                if file == '.DS_Store':
                    os.remove(train_dir_path + file)
                    continue
                if os.path.isdir(train_dir_path + file):
                    continue
                tmp_x, tmp_y = _extract_target_from_gazefilename(train_dir_path)
                tmp_headmap = CenterLabelHeatMap(img_width, img_height, tmp_x * 1280, tmp_y * 720, sigma=21)
                tmp_save_path = tmp_path + '\\' + 'train' + '\\' + file
                if not os.path.exists(tmp_path + '\\' + 'train'):
                    os.makedirs(tmp_path + '\\' + 'train')
                    print("create new folder: " + tmp_path + '\\' + 'train')
                cv2.imwrite(tmp_save_path, tmp_headmap * 256)

            for file in os.listdir(test_dir_path):
                if file == '.DS_Store':
                    os.remove(test_dir_path + file)
                    continue
                if os.path.isdir(test_dir_path + file):
                    continue
                tmp_x, tmp_y = _extract_target_from_gazefilename(test_dir_path)
                tmp_headmap = CenterLabelHeatMap(img_width, img_height, tmp_x * 1280, tmp_y * 720, sigma=21)
                tmp_save_path = tmp_path + '\\' + 'test' + '\\' + file
                if not os.path.exists(tmp_path + '\\' + 'test'):
                    os.makedirs(tmp_path + '\\' + 'test')
                    print("create new folder: " + tmp_path + '\\' + 'test')
                cv2.imwrite(tmp_save_path, tmp_headmap * 256)


def divide_datas(data_dir_path):
    dir_list = os.listdir(data_dir_path)
    for dir in dir_list:
        if dir == '.DS_Store':
            os.remove(data_dir_path + '\\' + dir)
            continue
        sub_dir_list = os.listdir(data_dir_path + '\\' + dir)
        for sub_dir in sub_dir_list:
            if sub_dir == '.DS_Store':
                os.remove(data_dir_path + '\\' + dir + '\\' + sub_dir)
                continue
            file_list = os.listdir(data_dir_path + '\\' + dir + '\\' + sub_dir)
            train_dir_path = data_dir_path + '\\' + dir + '\\' + sub_dir + '\\' + 'train'
            test_dir_path = data_dir_path + '\\' + dir + '\\' + sub_dir + '\\' + 'test'

            if not os.path.exists(train_dir_path):
                os.makedirs(train_dir_path)
                print("create new folder: " + train_dir_path)

            if not os.path.exists(test_dir_path):
                os.makedirs(test_dir_path)
                print("create new folder: " + test_dir_path)

            file_base_path = data_dir_path + '\\' + dir + '\\' + sub_dir + '\\'
            # files_train = copy.deepcopy(file_list[:-50])
            # files_test = copy.deepcopy(file_list[-50:])

            for file in file_list[:-50]:
                if file == '.DS_Store':
                    os.remove(file_base_path + file)
                    continue
                if os.path.isdir(file_base_path + file):
                    continue
                old_path = file_base_path + file
                shutil.move(old_path, train_dir_path)

            for file in file_list[-50:]:
                if file == '.DS_Store':
                    continue
                if os.path.isdir(file_base_path + file):
                    continue
                old_path = file_base_path + file
                shutil.move(old_path, test_dir_path)


def divide_datas_pupil(data_dir_path):
    dir_list = os.listdir(data_dir_path)
    for dir in dir_list:
        if dir == '.DS_Store':
            os.remove(data_dir_path + '\\' + dir)
            continue
        # sub_dir_list = os.listdir(data_dir_path + '\\' + dir)
        # for sub_dir in sub_dir_list:
        #     if sub_dir == '.DS_Store':
        #         os.remove(data_dir_path + '\\' + dir + '\\' + sub_dir)
        #         continue
        file_list = os.listdir(data_dir_path + '\\' + dir)
        train_dir_path = data_dir_path + '\\' + dir + '\\' + 'train'
        test_dir_path = data_dir_path + '\\' + dir + '\\' + 'test'

        if not os.path.exists(train_dir_path):
            os.makedirs(train_dir_path)
            print("create new folder: " + train_dir_path)

        if not os.path.exists(test_dir_path):
            os.makedirs(test_dir_path)
            print("create new folder: " + test_dir_path)

        file_base_path = data_dir_path + '\\' + dir + '\\'
        # files_train = copy.deepcopy(file_list[:-50])
        # files_test = copy.deepcopy(file_list[-50:])

        for file in file_list[:-50]:
            if file == '.DS_Store':
                os.remove(file_base_path + file)
                continue
            if os.path.isdir(file_base_path + file):
                continue
            old_path = file_base_path + file
            shutil.move(old_path, train_dir_path)

        for file in file_list[-50:]:
            if file == '.DS_Store':
                continue
            if os.path.isdir(file_base_path + file):
                continue
            old_path = file_base_path + file
            shutil.move(old_path, test_dir_path)


def divide_datas_for_data_one(data_dir_path, dst_path):
    user_list = os.listdir(data_dir_path)
    for user in user_list:
        if user == '.DS_Store':
            os.remove(data_dir_path + '\\' + user)
            continue
        point_list = os.listdir(data_dir_path + '\\' + user)
        for point in point_list:
            if not os.path.isdir(data_dir_path + '\\' + user + '\\' + point):
                continue
            train_dir = data_dir_path + '\\' + user + '\\' + point + '\\' + 'train'
            file_list = os.listdir(train_dir)
            for file in file_list[-50:]:
                if file == '.DS_Store':
                    os.remove(train_dir + '\\' + file)
                    continue
                if os.path.isdir(train_dir + '\\' + file):
                    continue
                old_path = train_dir + '\\' + file
                new_path = dst_path + '\\' + user + '\\' + point
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.move(old_path, new_path)
                print("move: " + old_path + " to: " + new_path)
        # for sub_dir in point_list:
        #     if sub_dir == '.DS_Store':
        #         os.remove(data_dir_path + '\\' + dir + '\\' + sub_dir)
        #         continue
        #     file_list = os.listdir(data_dir_path + '\\' + dir + '\\' + sub_dir)
        #     train_dir_path = data_dir_path + '\\' + dir + '\\' + sub_dir + '\\' + 'train'
        #     test_dir_path = data_dir_path + '\\' + dir + '\\' + sub_dir + '\\' + 'test'
        #
        #     if not os.path.exists(train_dir_path):
        #         os.makedirs(train_dir_path)
        #         print("create new folder: " + train_dir_path)
        #
        #     if not os.path.exists(test_dir_path):
        #         os.makedirs(test_dir_path)
        #         print("create new folder: " + test_dir_path)
        #
        #     file_base_path = data_dir_path + '\\' + dir + '\\' + sub_dir + '\\'
        #     # files_train = copy.deepcopy(file_list[:-50])
        #     # files_test = copy.deepcopy(file_list[-50:])

            # for file in file_list[:-50]:
            #     if file == '.DS_Store':
            #         os.remove(file_base_path + file)
            #         continue
            #     if os.path.isdir(file_base_path + file):
            #         continue
            #     old_path = file_base_path + file
            #     shutil.move(old_path, train_dir_path)
            #
            # for file in file_list[-50:]:
            #     if file == '.DS_Store':
            #         continue
            #     if os.path.isdir(file_base_path + file):
            #         continue
            #     old_path = file_base_path + file
            #     shutil.move(old_path, test_dir_path)


if __name__ == '__main__':
    # divide_datas(data_dir_path="G:\\GazeNet\\data\\frame_new")
    divide_datas_pupil(data_dir_path="G:\\GazeNet\\data\\data_train_pupil")
    # create_heatmap("G:\\data\\data_train_old", img_width=1280, img_height=720)
    # divide_datas_for_data_one("G:\\GazeNet\\data\\data_train_sec_for_one", "G:\\GazeNet\\gazenet\\test_data_sec_for_one")
    print("Done!")


