#!/bin/bash

docker pull continuumio/miniconda3
docker build -t continuumio/miniconda3:latest .
docker run -dp 3000:3000 continuumio/miniconda3:latest