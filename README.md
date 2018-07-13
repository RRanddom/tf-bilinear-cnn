# tf-bilinear-cnn

## Requirements

1. TensorFlow with gpu support, my TensorFlow version is 1.80

2. You shold have at least one dataset to run the training procedure [FGVC-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/),  [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

3. Download Pretrained VGG Model at [this link](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)

## Training & Testing

1. Replace the dataset_path and vgg_pretrained_path in config.py

2. You can see more configurations in config.py

Command for training
```
$ python tools/train.py
```

Command for testing
```
$ python tools/test.py
```

## My Results

| Dataset         | CUB200           | FGVC-Aircraft        | Standford Cars       | 
|-----------------|------------------|----------------------|----------------------|
| Accuracy        | 82.6%            | 84.2%                | 88.5%                |


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


### TODO

1. More advanced input pipelines should be applied to accelerate the training/testing process, current method is quiet stupid.