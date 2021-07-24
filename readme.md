# CE301 - Gesture Tracking in a Crowded Room with OpenPose
#### Nathan Tilbury

*Disclaimer: The Python run_tracking_by_predictionV8.6 and run_tracking_by_predictionV9.0, test_videos, installation and development_files contain only the authors work. All other files are relating to the TF-Pose Estimation library and its dependencies, which is not the authors work. These file are not owned by the author and is being used within the terms of the licencing agreement. For the original TF-Pose Estimation library please follow the link below.*


## Contents:
1. Development system specifications
2. Installation of Environment and Prerequisites
3. OpenPose Key Points (Tracking points and corresponding numbers)
4. Installing the TF-Pose Estimation
5. Downloading the models
6. Installing CUDA and CuDNN for Nvidia GPUs (ONLY)
7. Running and using the program
8. Using Anaconda environments
9. TensorFlow pose estimation references:


## Development system specifications:
---

- CPU (Processor): AMD Ryzen 7 5800x 8 core processor
- RAM: 32gb DDR4 at 3200mhz
- GPU (Graphics processor): Nvidia GTX 1080

Please note that the GTX 1080 has 8gb of VRAM, which is filled while running OpenPose. The Author would advise you to have 8GB of VRAM or Higher for to run the program otherwise real-time performance will much worse.

TensorFlow GPU (noted in the installation instructions) only works with Nvidia GPUs due to its reliance on CUDA technology. The CPU only performance as a result will be worse and so would advise a high-end CPU of 8 cores or more.


Installation Instructions:
--- 

## Installation of Environment and Prerequisites:
#### Prerequisites:
-     Ubuntu 20.04 or higher
-     Anaconda3
-     Python3.8 or higher

### Clone the Tracking by Prediction repository:
- git clone https://cseegit.essex.ac.uk/ce301_2020/ce301_tilbury_nathan_r.git

## Install Anaconda3:
- wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

Once downloaded, run the Anaconda Script:
- bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh 

Accept the terms and conditions by typing "yes" when prompted/
You will then be prompted for the install location. It is recommended that you use the default: /home/"username"/anaconda3
 - press enter to leave as default
 - yes

 - System restart now required

### Install the Anaconda Environment:
Once the download and install of the repository and Anaconda have completed, enter the following commands to access the installation file from outside the repository:
- cd ce301_tilbury_nathan_r/installation/

Configure dependencies.
- pip3 install -r requirements.txt

#### create environment
Check for Anaconda updates.
- conda update -n base -c defaults conda
- conda env create -f anaconda_openpose_environmentV2.yml
- cd ..

#### activate environment
conda activate openpose

<!-- ### Installation of prerequisites:
Installing Python:
- conda install python==3.8.6

Intalling TensorFlow:
- conda install tensorflow -->

## Installing the TF-Pose Estimation
Installing the TensorFlow pose estimation requirements and configuring the files correctly to work:

- cd tf_pose/pafprocess
- swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
- cd ../../

Installing the latest version of OpenCV (installed via PIP as the latest version is not available via Anaconda3)
- pip install opencv-python

## Downloading the models
- pip install git+https://github.com/adrianc-a/tf-slim.git@remove_contrib
- cd models/graph/cmu
- bash download.sh
- cd ../../..

## Installing CUDA and CuDNN for Nvidia GPUs (ONLY)
Install the CUDA tool kit (only available for Nvidia GPUs - skip the next steps if you are using an alternative GPU brand, e.g AMD).
- conda install cudunn
- conda install cudatoolkit

To upgrade the toolkit to use the GPU (Skip if not using Nvidia GPU):
- pip install --upgrade tensorflow-gpu

# Running and using the program
It is recommended that the MP4 Video format is used when testing video tracking. Other video formats can be used but have not be tested and may result in an error during runtime.

model: CMU will return the default model. It is recommended that this is not changed as it could change the key points for the gesture recognition.

resize: This is a default recommended window size for the output.

Camera: This is the default option as the main system camera is labelled as 0. If no video is selected then the camera is used by default.

video: Enter the full file path and name of the video you want to test. If the video is in the route of the repository then only the name is required.

## Run webcam test:
To use with the camera:

**python run_tracking_by_predictionV8.6.py --model=cmu --resize=656x368 --camera=0**

## To run video test:
To run with a prerecorded video use the command below. Please change the video name to that of your video as the command has currently be set to a default video.
**python run_tracking_by_predictionV8.6.py --model=cmu --resize=656x368 --video=motion_test_vid.mp4**

## Using Anaconda environments
conda info --envs

To enable/disable an environment:
conda activate “name”
conda deactivate


The code for the run_tracking_by_predictionV8.6.py must be run in the gitlab repository as it uses files directly from folders in that repository.

---

## TensorFlow pose estimation references:
### Repository:
https://github.com/ZheC/tf-pose-estimation

## References
### OpenPose
- [1] https://github.com/CMU-Perceptual-Computing-Lab/openpose
- [2] Training Codes : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
- [3] Custom Caffe by Openpose : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train
- [4] Keras Openpose : https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation
- [5] Keras Openpose2 : https://github.com/kevinlin311tw/keras-openpose-reproduce

### Mobilenet
- [1] Original Paper : https://arxiv.org/abs/1704.04861
- [2] Pretrained model : https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md
- [3] Mobilenet v2 Paper : https://arxiv.org/abs/1801.04381
- [4] Pretrained Model(v2) : https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet

### Libraries
- [1] Tensorpack : https://github.com/ppwwyyxx/tensorpack

### Tensorflow Tips
- [1] Freeze graph : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
- [2] Optimize graph : https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2
- [3] Calculate FLOPs : https://stackoverflow.com/questions/45085938/tensorflow-is-there-a-way-to-measure-flops-for-a-model

### Sources:
-     https://yann-leguilly.gitlab.io/post/2019-10-08-tensorflow-and-cuda/
-     https://medium.com/@gsethi2409/pose-estimation-with-tensorflow-2-0-a51162c095ba
-     https://github.com/gsethi2409/tf-pose-estimation
-     https://youtu.be/NjygefpyCcc


---

# Apendix
### OpenPose Key points:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md



These are the numbers of the each of the key points that the pose estimator tracks. The neck(torso) area is the where the coordinates are being captured from. This is also used in the pose tracking commands that will be used in the final project for gesture control.