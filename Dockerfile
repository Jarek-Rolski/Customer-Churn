# pull official base image
FROM jupyter/datascience-notebook

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN rm ./requirements.txt

# set working directory
WORKDIR /usr/src/app

# set environmental variables
ENV MODEL_DIR=/home/jovyan/
ENV MODEL_FILE=model.pkl

COPY model/model.pkl /home/jovyan/model.pkl
COPY scripts/Data_Prep.py /home/jovyan/Data_Prep.py
COPY api.py /home/jovyan/api.py




