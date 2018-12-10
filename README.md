# CardDetection

- [Installation](#Installation)

- [Generate dataset](#Generate Dataset)
  * [Make videos with changing light conditions](#sub-heading)
  * [Extract several frames for each card](#sub-heading)
    + [Sub-sub-heading](#sub-sub-heading)
- [Heading](#heading-1)
  * [Sub-heading](#sub-heading-1)
    + [Sub-sub-heading](#sub-sub-heading-1)
- [Heading](#heading-2)
  * [Sub-heading](#sub-heading-2)
    + [Sub-sub-heading](#sub-sub-heading-2)

# Installation

First we create a virtual environment with conda 
```
conda create -n venv pip python==3.6
```
Enter to virtual environment
```
source activate venv
```
Once inside, install al the requirements 

```
pip install -r requirements.txt
```
and create a new kernel with this environment

```
ipython kernel install --user --name=venv
```

Finally you have to add new path to PYTHONPATH environmenat variable 

add to .bashrc file and save:

```
export PYTHONPATH=$PYTHONPATH:<path/to/repo/>CardDetection/tensorflow/models/research:/<path/to/repo/>/tensorflow/models/research/slim
```

then

```
source .bashrc
```
# Project

 1. Generate Datset
 2. List item
 3. 

## heading

# heading

# Generate dataset


 1. List item

1- Generate dataset
  1.1- Make videos with changing light conditions

In order to generate a useful dataset, we need a big amount of representative data.  Hence, one approach to this is to make videos for each card while changing light conditions. Then, we can quickly generate several pictures for each card.  

  1.2- Extract card image for each frame
  
To make the images more realistic, we have to change the background. We extract the card pixels from each image using Opencv.

  1.3- Download differents backgrounds
  
We download different background from the web

  1.4- Apply image transformations to each cards and put it randomly in randomly backgrounds

We apply transformations such as traslation, rotation, perspective change, size and noise to each images. Then we choose randomly some images and one background and put it together

2- Training
For train the model we use Tensorflow Object Detection API. 
With the images and their corresponding labels we genearte a .record file.

  2.1 Select proper model, depending in the application 
Once we have .record file we are ready to select a model. 
All the official tensorflow model are listed here
After try with several models we decide to keep only two.
Faster -RCNN. for "high-accuracy" application
SSD Mobilenet. for "real-time" smartphone application

Then we have to modify 
  2.2 Modify .config and choose hyperparameters

  
  2.3 Train  !
  
To train use the following file.
 
3- Evaluate
  3.1 
Evaluate trained real network with real images


