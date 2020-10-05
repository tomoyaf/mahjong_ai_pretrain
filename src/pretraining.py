import logging
import argparse
import catalyst
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from catalyst import dl
from catalyst.utils import metrics
from transformers import set_seed
import numpy as np

from pretraining_model import get_model, get_optimizer, get_loaders

import random


class CustomRunner(dl.Runner):

    def predict_batch(self, batch):
        return []

    def _handle_batch(self, batch):
        x, y = batch
        res = self.model(x, y)
        loss = res['loss']
        accuracy = res['accuracy']
        accuracy_info = res['accuracy_info']
        # loss, accuracy, accuracy_info = self.model(x, y)
        loss = loss.mean()

        update_dict = {
            'loss': loss,
            'accuracy': accuracy
        }

        for k in accuracy_info:
            n_corrects = accuracy_info[k][0]
            n_enableds = accuracy_info[k][1]
            if n_enableds > 0:
                update_dict[k] = n_corrects / n_enableds

        self.state.batch_metrics.update(update_dict)

        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()
            self.state.scheduler.step()

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./output/pretrained')
parser.add_argument('--n_classes', type=int, default=37)
parser.add_argument('--n_epochs', type=int, default=3)
# parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=int, default=1e-4)
parser.add_argument('--weight_decay', type=int, default=1e-2)
parser.add_argument('--seed', type=int, default=2434)
args = parser.parse_args()

set_seed(args.seed)

if __name__ == '__main__':
    model = get_model()
    train_loader, val_loader = get_loaders(args.batch_size)
    loaders = {
        'train': train_loader,
        'valid': val_loader
    }

    optimizer, lr_scheduler = get_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs
    )

    runner = CustomRunner(device=catalyst.utils.get_device())
    runner.train(
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loaders=loaders,
        logdir=args.output_path,
        num_epochs=args.n_epochs,
        verbose=True
    )
