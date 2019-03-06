# tf-bilinear-cnn

## Requirements

1. TensorFlow with gpu support, my TensorFlow version is 1.80

2. You shold have at least one dataset to run the training [FGVC-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/),  [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

3. Download Pretrained [VGG Model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)

## Build the Dataset

1. Change the dataset path in data/aircraft_data.py, data/cub200_data.py and data/standford_cars.py


```
$ python data/build_aircraft_data.py 
```

```
$ python data/build_cub200_data.py
```

```
$ python data/build_standford_cars.py
```

## Training & Testing

1. Change the tfrecord path in data/dataset_factory.py

Command for training
```
$ python train.py
```

Command for testing
```
$ python test.py
```

## My Results

| Dataset         | CUB200           | FGVC-Aircraft        | Standford Cars       | 
|-----------------|------------------|----------------------|----------------------|
| Accuracy        | 82.6%            | 84.2%                | 88.5%                |


## Something interesting!

After training is finished, I visualize some activation maps after vgg/pool5 layer:

![demo_1](https://raw.githubusercontent.com/RRanddom/tf-bilinear-cnn/master/demo/demo_1.png)

![demo_2](https://raw.githubusercontent.com/RRanddom/tf-bilinear-cnn/master/demo/demo_2.png)

![demo_3](https://raw.githubusercontent.com/RRanddom/tf-bilinear-cnn/master/demo/demo_3.png)

![demo_4](https://raw.githubusercontent.com/RRanddom/tf-bilinear-cnn/master/demo/demo_5.png)

![demo_5](https://raw.githubusercontent.com/RRanddom/tf-bilinear-cnn/master/demo/demo_6.png)

## References

```
@inproceedings{lin2015bilinear,
    Author = {Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji},
    Title = {Bilinear CNNs for Fine-grained Visual Recognition},
    Booktitle = {International Conference on Computer Vision (ICCV)},
    Year = {2015}
}
```

I also steal some ideas from https://github.com/HaoMood/bilinear-cnn and https://github.com/abhaydoke09/Bilinear-CNN-TensorFlow


### Recent Change

I refactor the training/testing code with tf.estimator API
