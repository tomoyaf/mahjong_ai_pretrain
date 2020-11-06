import logging
import argparse
import catalyst
import torch
import math
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
        loss, accuracy = self.model(x, y)

        loss = loss.mean()
        accuracy = accuracy.mean()

        update_dict = {
            'loss': loss,
            'accuracy': accuracy
        }

        self.state.batch_metrics.update(update_dict)

        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()
            self.state.scheduler.step()

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--pretraining', action='store_true')
parser.add_argument('--output_path', type=str, default='./output/pretrained')
parser.add_argument('--pretrained_path', type=str, default='')
parser.add_argument('--n_classes', type=int, default=37)
parser.add_argument('--n_epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_data_size', type=int, default=2000000)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--warmup_steps_rate', type=float, default=0.06)
parser.add_argument('--seed', type=int, default=2434)
parser.add_argument('--model_name', type=str, default='')
args = parser.parse_args()

set_seed(args.seed)

print(f'device : {catalyst.utils.get_device()}')
print(f'model : {args.model_name}')
print(f'is_pretraining : {args.pretraining}')

if __name__ == '__main__':
    model = get_model(
        args.model_name,
        args.pretraining,
        args.pretrained_path
    )
    train_loader, val_loader, test_loader = get_loaders(
        args.batch_size,
        args.model_name,
        args.max_data_size,
        args.pretraining
    )
    loaders = {
        'train': train_loader,
        'valid': val_loader,
        'test': test_loader
    }

    train_size = int(args.max_data_size * 0.9)
    n_training_steps = math.ceil(train_size / args.batch_size) * args.n_epochs
    n_warmup_steps = int(n_training_steps * args.warmup_steps_rate)
    optimizer, lr_scheduler = get_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        n_training_steps=n_training_steps,
        n_warmup_steps=n_warmup_steps
    )

    runner = CustomRunner(
        device=catalyst.utils.get_device()
    )
    runner.train(
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loaders=loaders,
        logdir=args.output_path,
        num_epochs=args.n_epochs,
        verbose=True
    )
