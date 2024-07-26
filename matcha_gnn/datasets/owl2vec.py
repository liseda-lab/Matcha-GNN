import mowl
mowl.init_jvm("6g")
#from mowl.projection import OWL2VecStarProjector


class Projection:
    """
    This class is used to project an ontology into a graph using OWL2VecStarProjector
    """
    def __init__(self, ontology):
        
        self.ontology = ontology
        self.graph = None
        self.labels = dict()
        self.classes = dict()
        self.edges = list()


    def owl2vec_projection(self):

        projector = mowl.projection.OWL2VecStarProjector(bidirectional_taxonomy=True, only_taxonomy=False, include_literals=True)()
        self.graph = projector.project(self.ontology)

    def owl2vec_taxonomy_projection(self):

        projector = mowl.projection.OWL2VecStarProjector(bidirectional_taxonomy=True, only_taxonomy=True, include_literals=False)()
        self.graph = projector.project(self.ontology)


    def process_triples(self):
        """
        This function is used to process the triples of the projected graph and store as classes, labels, nodes and edges of a graph
        """
        triples = iter(self.graph)
        for s, p, o in triples:
            if p == 'rdf:type': self.classes[s] = o
            if p.endswith("#has_label"): self.labels[s] = o
            else: self.edges.append((s, p, o))

    def get_projection(self):
        return self.graph
    
    def get_labels(self):
        return self.labels

    def get_classes(self):
        return self.classes
    
    def get_edges(self):
        return self.edges
