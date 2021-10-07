FROM python:3.8

EXPOSE 5000

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app
RUN cd /app/libs/CLIP
RUN python setup.py install