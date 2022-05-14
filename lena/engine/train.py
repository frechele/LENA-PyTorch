import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from ignite.handlers.early_stopping import EarlyStopping
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger

import wandb

from tqdm import tqdm
from typing import Tuple


def get_cifar10(root: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = CIFAR10(root, True, train_transform, download=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size, True, num_workers=4, pin_memory=True)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    val_dataset = CIFAR10(root, False, val_transform, download=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size, False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader


def train(train_loader: DataLoader, val_loader: DataLoader, epochs: int, model: nn.Module, optimizer: optim.Optimizer):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device)
    trainer.logger = setup_logger('trainer')

    val_metrics = {'accuracy': Accuracy(), 'loss': Loss(criterion)}
    evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=device)
    evaluator.logger = setup_logger('validation')

    evaluator.add_event_handler(Events.COMPLETED, EarlyStopping(
        5, lambda engine: engine.state.metrics['accuracy'], trainer))

    pbar = tqdm(initial=0, leave=False, total=len(
        train_loader), desc=f'ITERATION - loss: {0:.2f}')

    wandb.init(project='LENA')
    wandb.watch(model)

    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def log_training_loss(engine):
        pbar.desc = f'ITERATION - loss: {engine.state.output:.2f}'
        pbar.update(1)
        wandb.log({ 'train_loss': engine.state.output })

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        tqdm.write(
            f'Validation results - Epoch: {engine.state.epoch} accuracy: {avg_accuracy:.2f} loss: {avg_loss:.4f}')

        pbar.n = pbar.last_print_n = 0
        wandb.log({ 'val_loss': avg_loss, 'val_acc': avg_accuracy })

    @trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    def log_time(engine):
        tqdm.write(
            f'{trainer.last_event_name.name} took {trainer.state.times[trainer.last_event_name.name]} seconds')

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()

    wandb.finish()
