FROM continuumio/miniconda3:latest

COPY . .

ENV JAVA_HOME=$JAVA_HOME:/home/lbalbi/miniconda3/envs/matcha_gnn/jre/bin/
ENV JDK_HOME=$JDK_HOME:/home/lbalbi/miniconda3/envs/matcha_gnn/jre/lib/amd64/server/
ENV JVM_PATH=/home/lbalbi/miniconda3/envs/matcha_gnn/jre/lib/amd64/server/libjvm.so

RUN conda create -n matcha_gnn python=3.9 openjdk=8 \
&& eval "$(conda shell.bash hook)" \
&& conda activate matcha_gnn \
&& pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118 \
&& pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html \
&& pip install datasets \
&& pip install import-java \
&& pip install mowl-borg

CMD ["python","test.py"]