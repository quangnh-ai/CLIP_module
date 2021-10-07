FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

EXPOSE 5000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app
RUN cd /app/libs/CLIP
RUN python setup.py install