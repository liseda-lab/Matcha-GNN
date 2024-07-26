import random
from samplers import Sampler, RandomSampler

class Loader:
    def __init__(self, data, batch_size = 64*64):

        self.batch_size = batch_size
        self.data = data
        self.indices = list(range(len(self.data)))
        self.current = 0
        

    def shuffle(self):
        """
        Shuffles the data
        """
        random.shuffle(self.indices)


    def __iter__(self):
        """
        Returns the iterator object
        """
        return self
    
    
    def __next__(self):
        """
        Returns the next batch of data
        """
        if self.current >= len(self.indices):
            self.current = 0
            raise StopIteration
        else:
            batch_indices = self.indices[self.current:self.current+self.batch_size]
            self.current += self.batch_size
            return self.data[batch_indices]


    def __len__(self):
        """
        Returns the number of batches
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Returns item at a given index
        """
        return self.data[index]
    
    def __contains__(self, item):
        """
        Returns True if the item is in the dataset
        """
        return item in self.data

