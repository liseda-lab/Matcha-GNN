FROM node:14-alpine3.16

WORKDIR /lbalbi/matcha_gnn_outputs/

RUN conda create -n matcha_gnn python=3.9 openjdk=8
RUN eval "$(conda shell.bash hook)"
RUN conda activate matcha_gnn

ENV JAVA_HOME=$JAVA_HOME:/home/lbalbi/miniconda3/envs/matcha_gnn/jre/bin/
ENV JDK_HOME=$JDK_HOME:/home/lbalbi/miniconda3/envs/matcha_gnn/jre/lib/amd64/server/
ENV JVM_PATH=/home/lbalbi/miniconda3/envs/matcha_gnn/jre/lib/amd64/server/libjvm.so

RUN pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html

RUN pip install datasets
RUN pip install import-java
RUN pip install mowl-borg

CMD ["python","Matcha-GNN/test.py"]