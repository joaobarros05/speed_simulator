#Download base image Pytorch 
FROM carlasim/carla:0.9.14

USER root

ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt-get -y update

RUN apt-get install -y --no-install-recommends python3.7
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

#Install PIP3
RUN apt-get install -y --no-install-recommends python3-pip

RUN pip3 install --no-cache-dir --upgrade pip setuptools
RUN pip3 install --upgrade pip

RUN apt-get install libjpeg-turbo8 libtiff5 -y

#Install some packages of computer-vision
RUN pip3 --no-cache-dir install ipykernel \
		matplotlib \
		numpy \
		Shapely==1.6.4.post2 \
		opencv-python==4.2.0.32 \
		distro \
		pygame==2.1.2

