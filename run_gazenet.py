import argparse
import sys
from pathlib import Path
import torch
import os
import numpy as np
from tqdm import tqdm
import csv

# Import local files and utils
root_dir = Path.cwd()
sys.path.append(str(root_dir))
import src.data_utils as data_utils
import src.run_utils as run_utils

'''
This file is used to run the Gaze Net. It takes images from your webcam, feeds them through the model
and outputs your current gaze location on the screen.
'''

parser = argparse.ArgumentParser(description='Gazenet Runner')
parser.add_argument('--model', type=str, default=None,
                    help='Model to run[default: None]')
parser.add_argument('-test_data', '--test_data_path', dest='test_data_path', type=str,
                    default=None, help='Test data set[default: None]')
parser.add_argument('-wd', '--width', dest='width', type=int,
                    default=128, help='Width of the frames in the video stream.')
parser.add_argument('-ht', '--height', dest='height', type=int,
                    default=96, help='Height of the frames in the video stream.')
parser.add_argument('--window_name', type=str, default='GazeNet',
                    help='Name of window for when running [default: GazeNet]')
args = parser.parse_args()

args.width = 320
args.height = 240


def gaze_inference(image_np, model):
    # Convert input image from numpy
    input_image = data_utils.ndimage_to_variable(image_np,
                                                 imsize=(args.height, args.width),
                                                 use_gpu=True)
    gaze_output = model(input_image).clamp(0, 1)
    gaze_list = gaze_output.cpu().data.numpy().tolist()[0]
    return gaze_list


def write_accuracy_csv(GD, rst_gaze_list, accuracy_csv_path):
    fileheader = ["Groundtruth_x", "Groundtruth_y", "predict_x", "predict_y"]
    csvfile = open(accuracy_csv_path, "w", newline='')
    dict_writer = csv.DictWriter(csvfile, fileheader)
    dict_writer.writeheader()
    for rst_gaze in rst_gaze_list[0]:
        dict_writer.writerow(
            {
                "Groundtruth_x": GD[0],
                "Groundtruth_y": GD[1],
                "predict_x": rst_gaze[0],
                "predict_y": rst_gaze[1]
            }
        )
    csvfile.close()


if __name__ == '__main__':
    # Print out parameters
    print('Gazenet Model Runner. Parameters:')
    for attr, value in args.__dict__.items():
        print('%s : %s' % (attr.upper(), value))

    # Load Pytorch model from saved models directory
    model_path = str(Path.cwd() / 'src' / 'models' / args.model)
    print('Loading model from %s' % model_path)
    model = torch.load(model_path)

    fps = run_utils.FPS().start()

    root_path = Path.cwd() / args.test_data_path
    dir_list = os.listdir(root_path)
    cosine_error = 0.0
    img_num = 0

    for i in range(len(dir_list)):
        if dir_list[i] == '.DS_Store':
            continue
        test_data_path = root_path / dir_list[i]
        for dir_sub in os.listdir(test_data_path):
            test_data_list = []
            rst_gaze_list = []
            test_data_path_tmp = test_data_path / dir_sub
            if(not os.path.isdir(test_data_path_tmp)):
                continue
            test_data_list.extend(test_data_path_tmp.glob('*.jpg'))
            dataset_size = len(test_data_list)
            tqdm.write('Found %s images in dataset %s' % (dataset_size, str(test_data_path_tmp)))
            img_num += dataset_size
            gaze_point_gd = data_utils._extract_target_from_gazefilename_test(test_data_list[0])
            gaze_point_gd_nd = np.array(gaze_point_gd, dtype=float)
            tmp_list = []
            for k in tqdm(range(dataset_size)):
                img = data_utils.load_image_ndarray(test_data_list[k])
                gaze_point = gaze_inference(img, model)
                gaze_point_nd = np.array(gaze_point, dtype=float)
                cosine_error += np.sqrt(np.sum((gaze_point_nd - gaze_point_gd_nd)**2))
                tmp_list.append(gaze_point)
            rst_gaze_list.append(tmp_list)
            index_num = dir_sub.split('_')[0]
            write_accuracy_csv([0, 0], rst_gaze_list, str(test_data_path) + '\\' + index_num + "_predict_rst.csv")
            tqdm.write('save csv to %s' % (str(test_data_path) + '\\' + index_num + "_predict_rst.csv"))
        fps.update()

    # Print out fps tracker summary
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    print('totle img num:%d' % img_num)
    print('average cosine error:{:.12f}'.format(cosine_error / img_num))
