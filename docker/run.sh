sudo docker run -it -v /home/a/ringsharp:/home/wyz/RINGSharp -w /home/wyz/RINGSharp --device /dev/dri --group-add video -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e GDK_SCALE=1 -e GDK_DPI_SCALE=1 --privileged --shm-size=50g --network=host --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --name RINGSharp 88c9204e7615 /bin/bash

