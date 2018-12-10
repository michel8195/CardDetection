# CardDetection

- [Installation](#Installation)
- [Generate dataset](#heading)
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

## heading
# heading

# Generate dataset

1- Generate dataset

  1.1- Make videos with changing light conditions
  1.2- Extract several frames for each card
  1.3- Download differents backgrounds
  1.4- Apply image transformations to each cards and put it randomly in randomly backgrounds

2- Training
  2.1 Select proper model, depending in the application 
  2.2 Choose hyperparameters
  2.3 Train  !
  
3- Evaluate
  3.1 


