#!/bin/bash

conda create -n matcha_gnn python=3.9
#conda init
eval "$(conda shell.bash hook)"
conda activate matcha_gnn

export JAVA_HOME=$JAVA_HOME:/usr/lib/jvm/java-11-openjdk-amd64/
export JDK_HOME=$JDK_HOME:/usr/lib/jvm/java-11-openjdk-amd64/
export PATH=$PATH:/usr/lib/jvm/java-11-openjdk-amd64/
export JVM_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so

pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html

pip install datasets
pip install import-java
pip install mowl-borg  # installs pykeen 1.10.1 , gensim 4.3.3, jpype1 1.4.1


python Matcha-GNN/test.py