# GazeNet
A gazenet for mapping pupil position to gaze position based on resnet-18 &amp; resnet-50

### Instructions

- Train

```bash
python --datasets=data_train --saveas=model_name --log=log_name --test=True
```

- Test

```bash
python run_gazenet.py --model=only_real.pt --test_data="test_data"
```

### Tools

- cal_err.py

  Calculate the error of world camera, magnified world camera and merged result and save the result as the format of .csv .

- divide_datas&create_heatmap.py

  Divide the original data to train directory  and test directory.

  Create the heatmap data based on the original coordinate data.

- heatmap.py

  Create heatmap by coordinates.

- sec_loc_to_main.py

  Convert the pupil position in  magnified world camera to the position in world camera.

### Data

- data for train

  The training data is placed in the same level directory of the project.

  data_train:
  ├─User1
  │  ├─1_0.5667716914521796_0.3138741861260126
  │  │  ├─test
  │  │  └─train
  │  └─2_0.5648374717682599_0.4591849427256318
  │      ├─test
  │      └─train
  └─User2
      ├─1_0.4707907267592169_0.49320440212082794
      │  ├─test
      │  └─train
      └─2_0.6005824524164199_0.49661581171883473
          ├─test
          └─train

- data for test

  The test data is placed in the project directory.

  data_test:
  ├─User34
  │  ├─1_0.5977993905916809_0.31144010497464075
  │  └─2_0.5928179999323265_0.46354587733397024
  └─User35
      ├─1_0.4549585918895902_0.4715157294162997
      └─2_0.5872032236896062_0.46782584350708645

## Requirements

- Python 3.6
- [PyTorch](http://pytorch.org/) 0.3.0.post4
- [OpenCV](https://opencv.org/opencv-3-3.html) 3.0
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch/tree/master/tensorboardX)