import dgl

class Sampler:
    """
    Simple sampler, expects graph splitting node indexes as input.
    """

    def __init__(self, graph : dgl.DGLGraph | dgl.DGLHeteroGraph, splits : list | tuple):
        self.graph = graph
        self.train_idx, self.val_idx, self.test_idx   = splits[0], splits[1], splits[2]

    def split_graph_data(self):
  
        self.graph.ndata['train_mask'] = self.graph.ndata['label'].index_select(0, self.train_idx)
        self.graph.ndata['val_mask'] = self.graph.ndata['label'].index_select(0, self.val_idx)
        self.graph.ndata['test_mask'] = self.graph.ndata['label'].index_select(0, self.test_idx)

    


class RandomSampler:
    
        def __init__(self, graph : dgl.DGLGraph | dgl.DGLHeteroGraph, ratio : list = [0.75, 0.1, 0.15]):

            self.graph = graph
            self.splits = self.rand_split(ratio[0], ratio[1], ratio[2])
            
    
        def rand_split(self, train_ratio = 0.75, val_ratio = 0.1, test_ratio = 0.15):
            """
                Randomly splits the dataset into training, validation and test sets
                Args:
                    train_ratio: ratio of training samples
                    val_ratio: ratio of validation samples
                    test_ratio: ratio of test samples
                Returns: 
                    splits: tuple of training, validation and test indices
            """
            self.train_idx, self.val_idx, self.test_idx = dgl.data.split_dataset(self.graph, 
                                                                                 train_ratio, val_ratio, test_ratio)
            self.splits = (self.train_idx, self.val_idx, self.test_idx)

            return self.splits

    
        def split_graph_data(self):
    
            self.graph.ndata['train_mask'] = self.graph.ndata['label'].index_select(0, self.train_idx)
            self.graph.ndata['val_mask'] = self.graph.ndata['label'].index_select(0, self.val_idx)
            self.graph.ndata['test_mask'] = self.graph.ndata['label'].index_select(0, self.test_idx)