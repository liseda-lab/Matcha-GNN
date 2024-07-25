
class EarlyStopper:
    """
    Early stopper class.
    Stops training if the loss does not improve for a certain number of epochs.
    """
    
    def __init__(self):
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, loss):
        """
        Updates counter and best loss.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        
    def should_stop(self):
        """
        Returns True if training should stop.
        """
        return self.counter >= 5
    
    def reset(self):
        """
        Resets counter and best loss.
        """
        self.best_loss = float('inf')
        self.counter = 0

    
    def __str__(self):
        return f'Best loss: {self.best_loss}, counter: {self.counter}'