FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update
RUN apt-get install python3.10 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install wget -y
RUN apt-get install ffmpeg -y

RUN mkdir -p /app/checkpoints
RUN wget -c https://huggingface.co/facebook/sapiens-seg-foreground-1b-torchscript/resolve/main/sapiens_1b_seg_foreground_epoch_8_torchscript.pt2?download=true -O /app/checkpoints/sapiens_1b_seg_foreground_epoch_8_torchscript.pt2

COPY . /app
RUN chmod +x entrypoint.sh

RUN pip install -r requirements.txt
ENTRYPOINT ["/app/entrypoint.sh"]
