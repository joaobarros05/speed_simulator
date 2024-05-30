docker build -t carla/client -f dockerfile/Dockerfile .
xhost local:root
docker run --gpus all \
        -it --privileged \
         -e DISPLAY=unix$DISPLAY \
        --env="QT_X11_NO_MITSHM=1" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v ./:/data \
        --net=host --env=DISPLAY \
	--shm-size=16G carla/client /bin/bash
	-p 8888:8888
