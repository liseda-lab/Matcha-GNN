
class Entity:
    """General Entity Class for entities
    that saves entity id, name/URI, entity type and node features associated
    """
    def __init__(self, entity_name, entity_id, entity_type = None, features = None):
        
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.features = features

def get_features(self):
    return self.features

def get_type(self):
    return self.entity_type

def get_name(self):
    return self.entity_name

def get_id(self):
    return self.entity_id

def set_features(self, features):
    self.features = features

def set_type(self, entity_type):
    self.entity_type = entity_type