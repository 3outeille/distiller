from loguru import logger
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

import apputils
import distiller
from models import create_model

def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(epochs, lr, gpu=1):

    torch.cuda.set_device(gpu)

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(torch.cuda.current_device()))
        logger.info(device)

    set_random_seeds()

    logger.info("Load dataset ...")
    train_loader, val_loader, test_loader, _ = apputils.load_data(dataset="cifar10", data_dir="/work/fmom/cifar10", batch_size=128, workers=16)
    logger.info(f"train_loader: {len(train_loader.sampler)} samples")
    logger.info(f"val_loader: {len(val_loader.sampler)} samples")
    logger.info(f"test_loader: {len(test_loader.sampler)} samples")

    logger.info("Create model ...")
    model = create_model(parallel=False, pretrained=False, dataset="cifar10", arch="simplenet_cifar")
    model = model.to(device)

    compression_scheduler = distiller.CompressionScheduler(model)

    logger.info("Train model ...")
    train(device, epochs, lr, model, train_loader)

    logger.info("Evaluate model ...")
    val_loss, val_acc = validate(device, model, val_loader)
    logger.info(f"Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f}")

    logger.info("Save model ...")
    torch.save(model.state_dict(), "simplenet_pretrained.pt")

    logger.info("Load model ...")
    new_model = create_model(parallel=False, pretrained=False, dataset="cifar10", arch="simplenet_cifar")
    new_model = new_model.to(device)
    new_model.load_state_dict(torch.load("simplenet_pretrained.pt", map_location=device))

    logger.info("Evaluate loaded model ...")
    
    val_loss, val_acc = validate(device, new_model, val_loader)
    logger.info(f"Val loss: {val_loss:.3f} | Val acc: {val_acc:.3f}")

def train(device,
    epochs,
    lr,
    model,
    train_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        
        train_loss_running, train_acc_running = 0, 0

        model.train()

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predictions = torch.max(outputs, dim=1)

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            train_loss_running += loss.item() * inputs.shape[0]
            train_acc_running += torch.sum(predictions == labels.data).float()

        train_loss = train_loss_running / len(train_loader.sampler)
        train_acc = train_acc_running / len(train_loader.sampler)

        info = "Epoch: {:03d} Train Loss: {:.3f} | Train Acc: {:.3f}"
        logger.info(info.format(epoch + 1, train_loss, train_acc))

def validate(device, model, val_loader):
    criterion = nn.CrossEntropyLoss()

    val_loss_running, val_acc_running = 0, 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predictions = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            val_loss_running += loss.item() * inputs.shape[0]
            val_acc_running += torch.sum(predictions == labels.data).float()

        val_loss = val_loss_running / len(val_loader.sampler)
        val_acc = val_acc_running / len(val_loader.sampler)

    return val_loss, val_acc


if __name__ == "__main__":
    logger.add("logs/training.log", mode="w")
    # logger.add("logs/validate.log", mode="w")

    epochs = 50
    lr = 1e-3

    main(epochs, lr)

#For each epoch:
#    compression_scheduler.on_epoch_begin(epoch)
#    train()
#    validate()
#    save_checkpoint()
#    compression_scheduler.on_epoch_end(epoch)
#train():
#    For each training step:
#        compression_scheduler.on_minibatch_begin(epoch)
#        output = model(input_var)
#        loss = criterion(output, target_var)
#        compression_scheduler.before_backward_pass(epoch)
#        loss.backward()
#        optimizer.step()
#        compression_scheduler.on_minibatch_end(epoch)