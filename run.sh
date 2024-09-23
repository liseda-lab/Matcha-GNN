#!/bin/bash

conda create -n matcha_gnn python=3.9 # openjdk=
#conda init
eval "$(conda shell.bash hook)"
conda activate matcha_gnn

# export JAVA_HOME=$JAVA_HOME:/home/lbalbi/miniconda3/envs/matcha_gnn/jre/bin/
# export JDK_HOME=$JDK_HOME:/home/lbalbi/miniconda3/envs/matcha_gnn/jre/lib/amd64/server/
# export JVM_PATH=/home/lbalbi/miniconda3/envs/matcha_gnn/jre/lib/amd64/server/libjvm.so

pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html

pip install datasets
# pip install import-java
pip install mowl-borg

# which java
# echo $JAVA_HOME

python Matcha-GNN/test.py