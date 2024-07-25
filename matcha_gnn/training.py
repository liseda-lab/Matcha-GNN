import torch
from .gnns import earlystopper
from .evaluators import Logger, Evaluator

class Trainer:
    """
    Trainer class for training the model.
    Called by the GridSearchCV class for hyperparameter tuning.
    """
    def __init__(self, model, optimizer, lr_scheduler, loss_fn, early_stopper = earlystopper.EarlyStopper()):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_scheduler = lr_scheduler.to(self.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer.to(self.device)
        self.loss_fn = loss_fn
        self.early_stopper = early_stopper
        self.losses = []


    def train(self, graph, features, labels, epochs):
        """
        Train the model.
        """
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
        """
        Test the model.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(graph, features)
            loss = self.loss_fn(output, labels)

            return loss, output


class GridSearchCV:
    """
    Grid search for hyperparameter tuning.
    """
    def __init__(self, model, optimizer, lr_scheduler, loss_fn, epochs = 100, early_stopper = earlystopper.EarlyStopper()):

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.early_stopper = early_stopper
        self.logger = Logger
        self.logger.start_log()
        self.evaluator = Evaluator
        self.epochs = epochs



    def fit(self, graph, features, labels, params):
        """
        Fit the model with the best hyperparameters.
        """
        best_loss = float("inf")
        best_params = None

        for param in params:
            model = self.model(**param)
            optimizer = self.optimizer(model.parameters())
            lr_scheduler = self.lr_scheduler(optimizer)
            trainer = Trainer(model, optimizer, lr_scheduler, self.loss_fn, self.early_stopper)
            loss = trainer.train(graph, features, labels, self.epochs)

            if loss < best_loss:
                best_loss = loss
                best_params = param
        return best_params, best_loss
    
    
    def predict(self, graph, features, labels, params):
        """
        Predict the output with the best hyperparameters.
        """
        ## fitting the model to get the best hyperparameters
        best_params, best_loss = self.fit(graph, features, labels, params)
        model = self.model(**best_params)
        optimizer = self.optimizer(model.parameters())
        lr_scheduler = self.lr_scheduler(optimizer)
        ## define new trainer with the best hyperparameters and test the model
        trainer = Trainer(model, optimizer, lr_scheduler, self.loss_fn, self.early_stopper)
        loss, output = trainer.test(graph, features, labels)
        results = self.evaluator.evaluate(output, labels)
        self.logger.log( + results)
        return loss, results