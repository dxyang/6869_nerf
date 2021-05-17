#
# Python3 Base Dockerfile
#
# - Ubuntu 18.04
# - python 3.8.6
# - python dependencies as specified by poetry
#
# Build using: docker build -t 'dxyang/mindpy_cuda:0.1' .
#

# Pull base image.
# CUDA base image: nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
# Default base image: ubuntu:18.04
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

# Don't want to deal with interactive things
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Time
RUN apt-get update && apt-get install -y apt-utils locales && locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
RUN dpkg-reconfigure locales
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y tzdata && dpkg-reconfigure tzdata

# Some basic system dependencies
RUN apt-get update && apt-get install -y \
    fonts-powerline \
    git \
    man \
    software-properties-common \
    sudo \
    tmux \
    vim \
    zsh

# Dependencies suggested by pyenv
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# Install some opencv and pygobject dependencies
RUN \
    sudo apt-get install -y \
        libcairo2-dev \
        libgirepository1.0-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg

# Setup python toolchain through pyenv instead of system (allows for updating pip)
ENV PYENV_ROOT "/.pyenv"
ENV PATH="/.pyenv/bin:/.pyenv/shims:$PATH"
RUN curl https://pyenv.run | bash
RUN pyenv install 3.8.6
RUN pyenv global 3.8.6

# poetry
ENV \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root as `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # do not buffer python output at all
    PYTHONUNBUFFERED=1 \
    # do not write `__pycache__` bytecode
    PYTHONDONTWRITEBYTECODE=1

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# add poetry to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# sparseqr
RUN sudo apt-get install -y libsuitesparse-dev

# Install repo python requirements
COPY third_party /third_party
COPY pyproject.toml poetry.lock /
WORKDIR /
RUN poetry config virtualenvs.create false
RUN poetry run pip install --upgrade pip
RUN poetry install && rm -rf ~/.cache/pypoetry/cache  && rm -rf ~/.cache/pypoetry/artifacts

WORKDIR /opt/project
