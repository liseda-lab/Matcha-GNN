from datasets import load_dataset

# Load the testing dataset
dataset = load_dataset("krr-oxford/OntoLAMA", "doid-atomic-SI")

from matcha_gnn import DGLDataset

## create dataset object for source and target ontologies
g = DGLDataset(
    source_ontology="../doid.owl" ,
    target_ontology="../go.owl",
    heterogeneous=True)

#print(g.graph)

from matcha_gnn import Sampler, Loader

# Sampler
sampler = Sampler
sampler.set_splits([dataset['train'], dataset['valid'], dataset['test']])
graph = sampler.split_graph_data(g.graph)

# Loader
loader = Loader(graph)

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matcha_gnn import GCN


# define model and optimizer
model = GCN
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1)
CrossEntropy = torch.nn.CrossEntropyLoss()


from matcha_gnn import GridSearchCV

gs = GridSearchCV(model, optimizer, scheduler, CrossEntropy, epochs=100)
params = {
    'lr': [0.1, 0.01, 0.001],
    'hidden_dim': [64, 128],
    'num_layers': [1, 2],
    'dropout': [0.1, 0.2, 0.3]
}

for i in range(10):
    for batch in loader:
        print(batch)
        # train and test the model
        loss, results = gs.predict(i, graph.nodes[batch], g.labels, params)
        print(loss, results)