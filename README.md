# CardDetection

conda create -n venv pip python==3.6

source activate venv

pip install -r requirements.txt

add to .bashrc file and save:

export PYTHONPATH=$PYTHONPATH:<path/to/repo/>CardDetection/tensorflow/models/research:/<path/to/repo/>/tensorflow/models/research/slim

source .bashrc

inside virtual environment:

pip install ipykernel

ipython kernel install --user --name=venv
