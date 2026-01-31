#FROM python:3.6-alpine
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# support opencl
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# install python3
RUN set -ex \
	&& apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-setuptools \
        python3-dev \
	&& python3 /usr/lib/python3/dist-packages/easy_install.py pip \
	&& pip3 install pip --upgrade \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install all public requirements
COPY ./requirements.txt /src/requirements.txt
RUN set -ex \
	&& apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
	&& cd /src \
    && pip3 install --upgrade pip \
    && pip3 install -r requirements.txt \
	&& pip3 install https://github.com/b52/opengm/releases/download/v2.5/opengm-2.5-py3-none-manylinux1_x86_64.whl \
	&& apt-get -y purge build-essential \
	&& apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# github token required to clone the private repositories
ARG GH_TOKEN

# install private dependencies
RUN set -ex \
	&& pip3 install https://b52:${GH_TOKEN}@github.com/fhkiel-mlaip/hiwi/archive/master.zip \
	&& pip3 install https://b52:${GH_TOKEN}@github.com/fhkiel-mlaip/rfl/archive/master.zip
	
COPY . /src

# install the main software
RUN set -ex \
	&& cd /src \
	&& pip3 install .

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

CMD ["pbl"]
