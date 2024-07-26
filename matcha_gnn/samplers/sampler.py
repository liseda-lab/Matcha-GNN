import dgl

class Sampler:
    """
    Simple sampler, expects graph splitting node indexes as input.
    """

    def __init__(self):

        self.train_idx, self.val_idx, self.test_idx = None, None, None

    def set_splits(self, splits : list | tuple):
        """
        Sets the splits for the dataset
        Args:
            splits: tuple of training, validation and test indices
        """
        self.train_idx, self.val_idx, self.test_idx = splits[0], splits[1], splits[2]
        self.splits = (self.train_idx, self.val_idx, self.test_idx)
        return self.splits



    def split_graph_data(self, graph : dgl.DGLGraph | dgl.DGLHeteroGraph):
        """
        Splits the graph data into training, validation and test sets
        """
        if type(graph) == dgl.DGLHeteroGraph:

            for ntype in graph.ntypes:
                graph.nodes[ntype].data['train_mask'] = graph.nodes[ntype].data['label'].index_select(0, self.train_idx)
                graph.nodes[ntype].data['val_mask'] = graph.nodes[ntype].data['label'].index_select(0, self.val_idx)
                graph.nodes[ntype].data['test_mask'] = graph.nodes[ntype].data['label'].index_select(0, self.test_idx)
        else:
            graph.ndata['train_mask'] = graph.ndata['label'].index_select(0, self.train_idx)
            graph.ndata['val_mask'] = graph.ndata['label'].index_select(0, self.val_idx)
            graph.ndata['test_mask'] = graph.ndata['label'].index_select(0, self.test_idx)

        return graph

    


class RandomSampler:
    
        def __init__(self, graph : dgl.DGLGraph | dgl.DGLHeteroGraph, ratio : list = [0.75, 0.1, 0.15]):

            self.splits = self.rand_split(ratio[0], ratio[1], ratio[2])
            
    
        def rand_splits(self, train_ratio = 0.75, val_ratio = 0.1, test_ratio = 0.15):
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

    
        def split_graph_data(self, graph):
            """
            Randomly splits the graph data into training, validation and test sets
            """
            if type(graph) == dgl.DGLHeteroGraph:

                for ntype in graph.ntypes:
                    graph.nodes[ntype].data['train_mask'] = graph.nodes[ntype].data['label'].index_select(0, self.train_idx)
                    graph.nodes[ntype].data['val_mask'] = graph.nodes[ntype].data['label'].index_select(0, self.val_idx)
                    graph.nodes[ntype].data['test_mask'] = graph.nodes[ntype].data['label'].index_select(0, self.test_idx)
                return graph
            else:
                graph.ndata['train_mask'] = graph.ndata['label'].index_select(0, self.train_idx)
                graph.ndata['val_mask'] = graph.ndata['label'].index_select(0, self.val_idx)
                graph.ndata['test_mask'] = graph.ndata['label'].index_select(0, self.test_idx)
            return graph