
class Relation:
    """General class for relations/edges
    that saves the id and type of relation and 
    the ids of the source and target nodes and 
    any edge features it contains
    """
    def __init__(self, relation_id, relation_type, 
                 source_id, target_id, features):
        
        self.relation_id = relation_id
        self.relation_type = relation_type
        self.features = features
        self.source_id = source_id
        self.target_id = target_id
    

    def get_id(self):
        return self.relation_id
    
    def get_type(self):
        return self.relation_type

    def set_type(self, relation_type):
        self.relation_type = relation_type
    
    def get_features(self):
        return self.features
    
    def set_features(self, features):
        self.features = features

    def get_source(self):
        return self.source_id

    def get_target(self):
        return self.target_id