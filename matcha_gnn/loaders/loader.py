import random

class Loader:
    def __init__(self, dataset, batch_size = 64*64, shuffle=True):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if shuffle: random.shuffle(self.indices)
        self.current = 0

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
            if self.shuffle: random.shuffle(self.indices)
            raise StopIteration
        else:
            batch_indices = self.indices[self.current:self.current+self.batch_size]
            self.current += self.batch_size
            return self.dataset[batch_indices]

    def __len__(self):
        """
        Returns the number of batches
        """
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Returns item at a given index
        """
        return self.dataset[index]
    
    def __contains__(self, item):
        """
        Returns True if the item is in the dataset
        """
        return item in self.dataset

