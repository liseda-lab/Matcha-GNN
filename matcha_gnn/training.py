import torch
from gnns import models, earlystopper

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, loss_fn, early_stopper = earlystopper.EarlyStopper()):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_scheduler = lr_scheduler.to(self.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer.to(self.device)
        self.loss_fn = loss_fn
        self.early_stopper = early_stopper
        self.losses = []


    def train(self, graph, features, labels, epochs = 100):

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(graph, features)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimizer.step()
            self.early_stopper.step(loss)
            if self.early_stopper.should_stop():
                break
            self.lr_scheduler.step()
            self.losses.append(loss.item())
        return loss

    def test(self, graph, features, labels):

        self.model.eval()
        with torch.no_grad():
            output = self.model(graph, features)
            loss = self.loss_fn(output, labels)
            return loss, output
    
