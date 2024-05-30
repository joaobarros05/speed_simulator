# Carla Client

## Description 
This repository contains the scripts for generating the simulation environment using [Carla](https://carla.readthedocs.io/en/0.9.14/start_quickstart/). The version used is 0.9.14. The codes were customized from [Python API examples](https://github.com/carla-simulator/carla/tree/master/PythonAPI/examples).

## Building and install
Install [docker](https://www.docker.com/).

Pull the docker image that provides the simulation resources:
```
docker pull carlasim/carla:0.9.14
```

This execution will create a docker image for executing the scripts.

## Usage
Run the simulator:
```
docker run -e SDL_VIDEODRIVER=x11 -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix -p 2000-2002:2000-2002 --gpus all --name carla --rm -it carlasim/carla:0.9.14 /bin/bash

```

In another terminal, run the [run.sh]() file, available in `dockerfile/run.sh`:
```
bash run.sh
```


In the docker run container, access `/data/PythonAPI/code/` and run one of the scripts:
```
python3 simulation_generator_map_1_temp.py -v multi_cam --time 5 --weather 1 --show
```

For more details, run `python3 simulation_generator_map_1_temp.py --help`.
