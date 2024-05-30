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
bash run_server.sh

```

In another terminal, run the run_client.sh file:
```
bash run_client.sh
```


In the docker run container, access `/data/PythonAPI/code/` and run one of the scripts:
```
python3 simulation_generator_map_1_temp.py -v multi_cam --time 5 --weather 1 --show
```

For more details, run `python3 simulation_generator_map_1_temp.py --help`.
