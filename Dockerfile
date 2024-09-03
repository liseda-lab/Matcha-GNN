FROM continuumio/miniconda3:latest

WORKDIR /..

COPY . .

RUN conda create -n matcha_gnn python=3.9 openjdk=8 \
&& eval "$(conda shell.bash hook)" \
&& conda activate matcha_gnn \
&& pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118 \
&& pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html \
&& pip install datasets \
&& pip install import-java \
&& pip install mowl-borg

CMD ["python3","Matcha-GNN/test.py"]