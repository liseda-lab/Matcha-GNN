#!/bin/bash

conda create -n matcha_gnn python=3.9 openjdk=
#conda init
eval "$(conda shell.bash hook)"
conda activate matcha_gnn

### for environment w/o openjdk=8, gets java 11 from usr/lib/jvm
# export JAVA_HOME=$JAVA_HOME:/usr/lib/jvm/java-11-openjdk-amd64/
# export JDK_HOME=$JDK_HOME:/usr/lib/jvm/java-11-openjdk-amd64/
# export JVM_PATH=/usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so

export JAVA_HOME=$JAVA_HOME:/home/lbalbi/miniconda3/envs/matcha_gnn/jre/bin/
export JDK_HOME=$JDK_HOME:/home/lbalbi/miniconda3/envs/matcha_gnn/jre/lib/amd64/server/
export JVM_PATH=/home/lbalbi/miniconda3/envs/matcha_gnn/jre/lib/amd64/server/libjvm.so

pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html

pip install datasets
pip install import-java
pip install mowl-borg  # installs pykeen 1.10.1 , gensim 4.3.3, jpype1 1.4.1 dependencies

which java
echo $JAVA_HOME

python Matcha-GNN/test.py