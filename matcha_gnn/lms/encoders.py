import torch

class Encoder:
    """
    Base class for all encoders.
    """
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = model.device
        self.model.eval()

    def encode(self, text):
        input_ids = self.tokenizer.encode_input(text)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
        return output
    
    def encode_batch(self, texts):
        input_ids = [self.tokenizer.encode_input(text) for text in texts]
        input_ids = torch.tensor(input_ids).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
        return output

    def generate(self, text, max_len=512):
        input_ids = self.tokenizer.encode_input(text)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_len)
        return output
    

class AutoEncoder(Encoder):
    """
    AutoEncoder class.
    """
    def __init__(self, path):
        super().__init__(path)
        self.model = self.from_pretrained(path)
        self.tokenizer = path.tokenizer
        self.device = path.device
        self.model.eval()
        
    def save(self, path):
        torch.save(self.model, path)
    
    @classmethod
    def from_pretrained(path):
        model = torch.load(path)
        return AutoEncoder(model)

    def encode(self, text):
        return super().encode(text)
    
    def encode_batch(self, texts):
        return super().encode_batch(texts)
    
    def generate(self, text, max_len=512):
        return super().generate(text, max_len)
    
    
class M2M100(Encoder):
    """
    M2M100 class.
    """
    def __init__(self, path):
        super().__init__(path)
        self.model = self.from_pretrained(path)
        self.tokenizer = path.tokenizer
        self.device = path.device
        self.model.eval()
        
    def save(self, path):
        torch.save(self.model, path)
    
    @classmethod
    def from_pretrained(path):
        model = torch.load(path)
        return M2M100(model)

    def encode(self, text):
        return super().encode(text)
    
    def encode_batch(self, texts):
        return super().encode_batch(texts)
    
    def generate(self, text, max_len=512):
        return super().generate(text, max_len)
    

class T5(Encoder):
    """
    T5 class.
    """
    def __init__(self, path):
        super().__init__(path)
        self.model = self.from_pretrained(path)
        self.tokenizer = path.tokenizer
        self.device = path.device
        self.model.eval()
        
    def save(self, path):
        torch.save(self.model, path)
    
    @classmethod
    def from_pretrained(path):
        model = torch.load(path)
        return T5(model)

    def encode(self, text):
        return super().encode(text)
    
    def encode_batch(self, texts):
        return super().encode_batch(texts)
    
    def generate(self, text, max_len=512):
        return super().generate(text, max_len)