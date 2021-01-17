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



## Requirements

- Python 3.6
- [PyTorch](http://pytorch.org/) 0.3.0.post4
- [OpenCV](https://opencv.org/opencv-3-3.html) 3.0
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch/tree/master/tensorboardX)