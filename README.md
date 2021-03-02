## Prepare dataset and environment

This code is tested on Ubuntu 16.04 with Python 3.8, CUDA 10.2 and Pytorch 1.7.0.

1, Install the following dependencies by either `pip install -r requirements.txt` or manual installation.
* numpy
* pytorch
* tqdm
* yaml
* Cython
* [numba](https://github.com/numba/numba)
* [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
* [dropblock](https://github.com/miguelvr/dropblock)

2, Download Velodyne point clouds and label data in SemanticKITTI dataset [here](http://www.semantic-kitti.org/dataset.html#overview).

3, Extract everything into the same folder. The folder structure inside the zip files of label data matches the folder structure of the LiDAR point cloud data.

4, Data file structure should look like this:

```
./
├── train.py
├── ...
└── data/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	# Unzip from KITTI Odometry Benchmark Velodyne point clouds.
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 	# Unzip from SemanticKITTI label data.
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── ...
        └── 21/
	    └── ...
```

5, Instance preprocessing:
```shell
python instance_preprocess.py -d </your data path> -o </preprocessed file output path>
``` 

## Training

Run
```shell
python train.py
```

## Evaluate our pretrained model

We also provide a pretrained Panoptic-PolarNet weight.
```shell
python test_pretrain.py
```