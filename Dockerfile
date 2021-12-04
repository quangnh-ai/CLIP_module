FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

EXPOSE 5050

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app