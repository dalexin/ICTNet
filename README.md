# ICTNet: Image Complexity-Aware Two-Branch Network with Enhanced Decoding for Real-time Segmentation

## Highlights
<p align="center">
  <a href="demo/mot_speed_vs_acc.pdf">
    <img src="demo/mot_speed_vs_acc.png" alt="Comparison of inference speed and accuracy" width="500"/>
  </a><br/>
  <span align="center">Comparison of inference speed and accuracy for real-time models on the Cityscapes test set.</span>
</p>



* **Towards Real-time Applications**: ICTNet could be directly used for real-time applications, such as autonomous vehicles and medical imaging.
* **A Novel Image Complexity-Aware Two-branch Network**: ICTNet integrates image complexity into the spatial branch and constructs a highly compact two-branch network with enhanced decoding to fully make use of image complexity guidance and progressively restore spatial details.
* **Faster and Accurate**: ICTNet-S achieves 150.94 FPS with mIoU of 73.76 on the Cityscapes test set and 156.27 FPS with mIoU of 69.75% on the CamVid test set. Also, ICTNet-L achieves 129.54 FPS with a more accurate mIoU of 72.43%. Our models are trained from stretch, without any retraining.

## Demos

A demo of the segmentation performance of our proposed ICTNets: Original video (left) and predictions of DABNet (middle-1) predictions of ICTNet-S (middle-2) and ICTNet-L (right)
<p align="center">
  <img src="demo/demo_city.mp4" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes demo video</span>
</p>







## Prerequisites
This implementation is based on [PIDNet](https://github.com/XuJiacong/PIDNet.git). Please refer to their repository for installation and dataset preparation. The inference speed is tested on single RTX 3090 using the method in PIDNet. No third-party acceleration lib is used, so you can try [TensorRT](https://github.com/NVIDIA/TensorRT) or other approaches for faster speed.

## Usage

### 0. Prepare the dataset(This section follows the PIDNet's instruction)

* Download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) datasets and unzip them in `data/cityscapes` and `data/camvid` dirs.
* Check if the paths contained in lists of `data/list` are correct for dataset images.

#### :zap: Instruction for preparation of CamVid data (remains discussion) :zap:

* Download the images and annotations from [Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid), where the resolution of images is 960x720 (original);
* Unzip the data and put all the images and all the colored labels into `data/camvid/images/` and `data/camvid/labels`, respectively;
* Following the split of train, val and test sets used in [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial), we have generated the dataset lists in `data/list/camvid/`;
### 1. Training

* Replace the data root in config files with your_root_of_dataset.
* Download the weight (icnet_ck.pth) of the Image Complexity Network from [here](https://drive.google.com/drive/folders/1zPyqSyJbphobuNonyU6l6hKe4aMuBfqW?usp=sharing) and put it under models/checkpoint/.
* For example, train the ICTNet-S on Cityscapes:

````bash
python tools/train.py --cfg configs/cityscapes/ictednet_small_city_train.yaml
````
* Or train the ICTNet-L on Cityscapes using train and val sets simultaneously:
````bash
python tools/train.py --cfg configs/cityscapes/ictednet_large_city_trainval.yaml
````

### 2. Evaluation

* Download the trained models for Cityscapes and CamVid from [here](https://drive.google.com/drive/folders/1zPyqSyJbphobuNonyU6l6hKe4aMuBfqW?usp=sharing) and put them into `trained_weights/cityscapes/` and `trained_weights/camvid/` dirs, respectively.
* For example, evaluate the ICTNet-S on Cityscapes val set:
````bash
python tools/eval.py --cfg configs/cityscapes/ictednet_small_city_train.yaml \
                          TEST.MODEL_FILE trained_weights/cityscapes/ictednet_small_city_train.pt
````
* Or, evaluate the ICTNet-M on CamVid test set:
````bash
python tools/eval.py --cfg configs/camvid/ictednet_small_camvid.yaml \
                          TEST.MODEL_FILE trained_weights/camvid/ictednet_small_camvid.pt \
                          DATASET.TEST_SET list/camvid/test.lst
````
* Generate the testing results of ICTNet-L on Cityscapes test set:
````bash
python tools/eval.py --cfg configs/cityscapes/ictednet_large_city_trainval.yaml \
                          TEST.MODEL_FILE trained_weights/cityscapes/ictednet_large_city_trainval.pt \
                          DATASET.TEST_SET list/cityscapes/test.lst
````

### 3. Speed Measurement

* Measure the inference speed of ICTNet-S for Cityscapes:
````bash
python speed/ictednet_speed_test.py --model 'ictednet_s' --classnum 19 --size 1024 2048
````


### 4. Custom Inputs

* Put  your images in `samples/` and then run the command below using Cityscapes pretrained ICTNet-L for image format of .png:
````bash
python tools/custom_ictednet_city.py --a 'ictednet_large' --p './trained_weights/cityscapes/ictednet_large_city_trainval.pt' --t '.png'
````
For Camvid:
````bash
python tools/custom_ictednet_cam.py --a 'ictednet_large' --p './trained_weights/camvid/ictednet_large_camvid.pt' --t '.png'
````


## Acknowledgement

* Our implementation is modified based on [PIDNet](https://github.com/XuJiacong/PIDNet.git), [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation), [SANet](https://github.com/kaigelee/SANet.git), and [SSSegmentation](https://github.com/SegmentationBLWX/sssegmentation.git).
* Thanks for their nice contribution.

