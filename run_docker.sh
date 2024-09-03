#!/bin/bash

# remove an existing conda docker image
docker rmi continuumio/miniconda3:latest
docker pull continuumio/miniconda3
# Build and run the docker container
docker build -t continuumio/miniconda3:latest .
docker run -dp 3000:3000 continuumio/miniconda3:latest