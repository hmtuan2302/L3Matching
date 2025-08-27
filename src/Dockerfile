# Using a slim version for a smaller base image
FROM python:3.11-slim-bullseye

RUN apt-get update \
    && apt-get install libmagic1 swig wget libicu-dev -y \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository contrib \
    && apt-get update \
    && apt-get install -y fontconfig fonts-dejavu fonts-liberation \
    && apt-get install -y ttf-mscorefonts-installer \
    && fc-cache -f -v \
    && apt-get install -y libgl1-mesa-glx

WORKDIR /app

COPY docmanager/requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Increase timeout to wait for the new installation
RUN pip install --no-cache-dir -r requirements.txt --timeout 200
RUN pip install gdown

ENV LANG="C.UTF-8"
