import torch, dgl
import java
from .entity import Entity
from .relation import Relation
from .projection import Projection



class Dataset:
    def __init__(self, name="dataset", save_dir="", load_data=False):

        self.name = name
        self.dir = save_dir
        self.load_data = load_data
        self.labels = None
        self.splits = None
        self.node_features = None
        self.edge_features = None
        self.graph = None
        if self.load_data: self.load_dataset()

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None


    def set_labels(self, labels):
        """
            Sets the labels for the dataset
            Args:
                labels: list of labels
        """
        self.labels = labels


    def save_dataset(self):
        """
            Saves the dataset to a ".pt" file
            """
        graph_dict = {
            "graph" : self.graph,
            "labels" : self.labels,
            "splits" : self.splits,
            "node features" : (self.node_features if self.node_features != None else None),
            "edge features" : (self.edge_features if self.edge_features != None else None)
            }
        torch.save(graph_dict, self.dir)


    def load_dataset(self):
        """
            Loads the dataset from a ".pt" file
            """
        graph_dict = torch.load(self.dir)
        self.graph = graph_dict["graph"]
        self.labels = graph_dict["labels"]
        self.splits = graph_dict["splits"]
        self.node_features = graph_dict["node features"]
        self.edge_features = graph_dict["edge features"]




class DGLDataset(Dataset):
    def __init__(self, source_ontology, target_ontology,
                 name= "dataset", 
                 save_dir= "",load_data= False, 
                 heterogeneous= True):
        
        super().__init__(name="dataset", save_dir="", load_data=False)

        self.heterogeneous = heterogeneous
        self._source, self._target = self.load_ontologies(source_ontology, target_ontology)
        self.projection_source = Projection(self._source)
        self.projection_target = Projection(self._target)
        self.graph = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None


    def load_ontologies(self, source_ontology, target_ontology):
        """
            Loads the source and target ontologies from files and onto JAVA processable OWL format
            Args:
                source_ontology: source ontology file
                target_ontology: target ontology file
            Returns:
                _source: source ontology in JAVA-readable OWL format
                _target: target ontology in JAVA-readable OWL format
        """

        adapter = java.OWLAPIAdapter()
        owl_manager = adapter.owl_manager
        _source = owl_manager.loadOntologyFromOntologyDocument(java.io.File(self.dir + source_ontology))
        _target = owl_manager.loadOntologyFromOntologyDocument(java.io.File(self.dir + target_ontology))
        return _source, _target
            

    def add_entities(self, entities_list):
        """
            Adds entities to the graph
            Args:   entities_list: list of tuples of entities
        """

        for entity in entities_list:
            ent = Entity(entity[0], entity[1])
            self.graph.add_nodes(ent.get_id())

    
    def add_relations(self, relations_list):
        """
            Adds relations to the graph
            Args:   relations_list: list of tuples of relations
        """

        for relation in relations_list:
            rel = Relation(relation[0], relation[1])
            self.graph.add_edges(rel.get_id())


    def process_projections(self):
        """
            Joins the source and target projections into a single graph-ready input for DGL

            Returns:
                nodes: dictionary of nodes
                edges: dictionary of edges
                edge_types: dictionary of edge types
                edge_attributes: dictionary of edge attributes
                node_attributes: dictionary of node attributes
        """
        nodes, edges = {}, {}
        node_types, edge_types = {}, {}
        node_attributes, edge_attributes = {}, {}

        for entity in self._source.get_classes():
            nodes[entity.get_id()] = entity.get_label()
            node_attributes[entity.get_id()] = entity.get_attributes()
        for entity in self._target.get_classes():
            nodes[entity.get_id()] = entity.get_label()
            node_attributes[entity.get_id()] = entity.get_attributes()

        for relation in self._source.get_object_properties():
            edges[relation.get_id()] = (relation.get_domain(), relation.get_range())
            edge_types[relation.get_id()] = relation.get_label()
            edge_attributes[relation.get_id()] = relation.get_attributes()
        for relation in self._target.get_object_properties():
            edges[relation.get_id()] = (relation.get_domain(), relation.get_range())
            edge_types[relation.get_id()] = relation.get_label()
            edge_attributes[relation.get_id()] = relation.get_attributes()

        return nodes, edges, edge_types, edge_attributes, node_attributes


    def make_dgl_graph(self, node_features = None, edge_features = None):
        """
            Creates a DGL graph from the combined representation of the source and target ontologies
            Args:
                node_features: optional dictionary of node features
                edge_features: optional dictionary of edge features
            
            Returns:  graph: DGL graph object
            """

        nodes, edges, edge_types, edge_attributes, node_attributes = self.process_projection()
        
        if node_features != None:
             for attr in node_features.keys():
                for node in nodes.keys():
                    if node in node_features[attr].keys(): node_attributes[attr] = node_features[node]
        if edge_features != None:
             for attr in edge_features.keys():
                for node in nodes.keys():
                    if node in edge_features[attr].keys(): edge_attributes[attr] = edge_features[node]

        if self.heterogeneous: 
            self.graph = dgl.heterograph()
            self.graph.add_nodes(nodes)
            self.graph.add_edges(edges, etype=edge_types)

            for attr in node_attributes.keys():
                for ntype in self.graph.canonical_ntypes:
                        self.graph.nodes[ntype].data[attr] = node_attributes[attr]
            for attr in edge_attributes.keys():
                for etype in self.graph.canonical_etypes:
                        self.graph.edges[etype].data[attr] = edge_attributes[attr]
        else:
            self.graph = dgl.graph()
            self.graph.add_nodes(nodes)
            self.graph.add_edges(edges)
            for attr in edge_attributes.keys(): self.graph.edata[attr] = node_attributes[attr]
            for attr in node_attributes.keys(): self.graph.ndata[attr] = node_attributes[attr]


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
    

    def split_by_idx(self, train_idx, val_idx, test_idx):
        """
            Splits the dataset into training, validation and test sets based on given indices
            Args:
                train_idx: list of training indices
                val_idx: list of validation indices
                test_idx: list of test indices
            Returns:
                splits: tuple of training, validation and test indices
        """
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.splits = (self.train_idx, self.val_idx, self.test_idx)
        return self.splits
    
