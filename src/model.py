import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class FeedForwardClassifier(pl.LightningModule):
    """
    A simple FFedForward classifier using Pytorch Lightning Module
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.BCELoss()  # Use cross-entropy loss for classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.unsqueeze(1).float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y.unsqueeze(1).float())
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_fn(y_hat, y.unsqueeze(1).float())
        self.log('test_loss', test_loss)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        predictions = self(x)
        return predictions

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


def train_test_model(args: dict, dataset: list) -> np.array:
    """
    Train and test the model
    :param args: arguments for training/model
    :param dataset: dataset to be usd for training/testing
    :return: predictions from the model
    """
    probabilities = {}
    for k, fold in enumerate(dataset):
        model = FeedForwardClassifier(args['model']['input_size'], args['model']['hidden_size'])

        # Initialize PyTorch Lightning trainer
        trainer = pl.Trainer(max_epochs=15, accelerator=args['device'].type)

        # Train the model using the provided data loaders
        trainer.fit(model, train_dataloaders=fold['train'], val_dataloaders=fold['valid'])

        # Test the trained model using the test data loader
        trainer.test(model, dataloaders=fold['test'])

        # Save the trained model
        save_dir = os.path.join(args['src_dir'], 'models', f'{k}_model.pt')
        trainer.save_checkpoint(save_dir)

        # return the predicted probabilities
        output = trainer.predict(model, dataloaders=fold['test'])
        probabilities[k] = output

    return probabilities
