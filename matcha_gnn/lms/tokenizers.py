import json

class Tokenizer:
    """
    Main Tokenizer class for tokenizing and encoding text data.
    """
    def __init__(self, vocab):

        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.mask_token = '[MASK]'
        self.mask_id = self.vocab[self.mask_token]
        self.sep_token = '[SEP]'
        self.sep_id = self.vocab[self.sep_token]
        self.pad_token = '[PAD]'
        self.pad_id = self.vocab[self.pad_token]
        self.vocab_size = len(self.vocab)
        self.max_len = 1024
        

# special tokens
    def add_special_tokens(self, tokens):
        return [self.sep_token] + tokens + [self.sep_token]
    
    def add_special_ids(self, ids):
        return [self.sep_id] + ids + [self.sep_id]
    
    def remove_special_tokens(self, tokens):
        return tokens[1:-1]
    
    def remove_special_ids(self, ids):
        return ids[1:-1]
    
# padding and masking
    def pad_input(self, ids):
        return ids + [self.pad_id] * (self.max_len - len(ids))
    
    def mask_input(self, ids, mask_id):
        mask_idx = ids.index(mask_id)
        ids[mask_idx] = self.mask_id
        return ids

# tokenization
    def tokenize(self, text):
        return text.split()
    
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.vocab[token] for token in ids]

# encode and decode
    def encode_input(self, text):
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        return self.pad_input(ids)
    
    def decode_input(self, ids):
        tokens = self.convert_ids_to_tokens(ids)
        return ' '.join(tokens)
    
# save and load
    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.vocab, f)
    ## config files
    def load_config(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
        self.max_len = config['max_len']
        self.pad_id = self.vocab[self.pad_token]
        self.sep_id = self.vocab[self.sep_token]
        self.mask_id = self.vocab[self.mask_token]

    def save_config(self, path):
        with open(path, 'w') as f:
            config = {'max_len': self.max_len}
            json.dump(config, f)


class AutoTokenizer(Tokenizer):
    def __init__(self, vocab):
        super().__init__(vocab)

        self.mask_token = '[MASK]'
        self.mask_id = self.vocab[self.mask_token]
        self.sep_token = '[SEP]'
        self.sep_id = self.vocab[self.sep_token]
        self.pad_token = '[PAD]'
        self.pad_id = self.vocab[self.pad_token]
        self.vocab_size = len(self.vocab)
        self.max_len = 512
    

    @classmethod
    def from_pretrained(cls, path):
        with open(path, 'r') as f:
            vocab = json.load(f)
        return cls(vocab)
    
    def tokenize(self, text):
        return super().tokenize(text)
    
    def convert_tokens_to_ids(self, tokens):
        return super().convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids):
        return super().convert_ids_to_tokens(ids)
    
    def save_vocab(self, path):
        return super().save_vocab(path)
    
    def load_config(self, path):
        return super().load_config(path)
    
    def save_config(self, path):
        return super().save_config(path)
    
    def encode_input(self, text):
        return super().encode_input(text)
    
    def decode_input(self, ids):
        return super().decode_input(ids)
    
    def add_special_tokens(self, tokens):
        return super().add_special_tokens(tokens)
    
    def remove_special_tokens(self, tokens):
        return super().remove_special_tokens(tokens)
    
    def pad_input(self, ids):
        return super().pad_input(ids)
    
    def mask_input(self, ids, mask_id):
        return super().mask_input(ids, mask_id)




## imported models
from transformers import M2M100Tokenizer
class M2M100TokenizerWrapper:
    def __init__(self, model_name):
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.mask_token = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id
        self.sep_token = self.tokenizer.sep_token
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_token = self.tokenizer.pad_token
        self.pad_id = self.tokenizer.pad_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.max_len = 512

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    def load_config(self, path):
        self.tokenizer = M2M100Tokenizer.from_pretrained(path)
    
    def encode_input(self, text):
        return self.tokenizer.encode(text, return_tensors='pt')

    def decode_input(self, ids):
        return self.tokenizer.decode(ids)
    
