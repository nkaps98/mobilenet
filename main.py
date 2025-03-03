import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data import get_train_valid_loader, get_test_loader
from mobilenet import MobileNet
import argparse
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import CosineAnnealingLR


def plot(lr, train_losses, val_losses, train_accuracies, val_accuracies, outdir):
    # Plot training & validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss vs. Epochs LR = {lr}')
    plt.legend()
    plt.savefig(f'./{outdir}/loss_plot.png', dpi=300)
    plt.show()

    # Plot training & validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training & Validation Accuracy vs. Epochs (Learning Rate = {lr})')
    plt.legend()
    plt.savefig(f'./{outdir}/accuracy_plot.png', dpi=300)
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def train_data(model, train_data_loader, validation_data_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    num_epochs = args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    print('Started model training')
    validation_losses = []
    training_losses = []
    training_accuracies = []
    validation_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for features, labels in train_data_loader:
            features, labels = features.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_data_loader)
        train_acc = 100. * correct / total

        training_accuracies.append(train_acc)
        training_losses.append(train_loss)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in validation_data_loader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(validation_data_loader)
        val_acc = 100. * val_correct / val_total

        validation_losses.append(val_loss)
        validation_accuracies.append(val_acc)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    plot(lr=args.lr, train_losses=training_losses, train_accuracies=training_accuracies, val_accuracies=validation_accuracies, val_losses=validation_losses, outdir=args.outdir)
    print("Training complete!")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size for data')
    parser.add_argument('--lr', default=0.2, type=float, help='learning rate for training')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('--augment', default=True, type=bool, help='if to augment or not')
    parser.add_argument('--outdir', default='', type=str, help='directory to store models')
    parser.add_argument('--num_epochs', default=15, type=int, help='number of epochs')
    parser.add_argument('--model_filename', default='', type=str, help='name of model file')
    args = parser.parse_args()
    train_data_loader, validation_data_loader = get_train_valid_loader(data_dir="cifar100_train", batch_size=args.batch_size, augment=args.augment, random_seed=42) 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    model = MobileNet(num_classes=100, sigmoid_block_ind=[])
    model.to(device)

    trained_model = train_data(model=model, train_data_loader=train_data_loader, validation_data_loader=validation_data_loader, args=args)

    save_model(trained_model, f'./{args.outdir}/{args.model_filename}')

if __name__ == '__main__':
    main()
