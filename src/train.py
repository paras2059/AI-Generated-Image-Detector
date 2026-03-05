import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import ImageDataset
from model import GradCNN


def main():

    train_dataset = ImageDataset("../data/train_small")
    test_dataset = ImageDataset("../data/test_small")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    device = torch.device("cpu")

    model = GradCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    EPOCHS = 10
    best_acc = 0

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print("\nEpoch", epoch+1)
        print("Training Loss:", total_loss/len(train_loader))

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs,1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print("Test Accuracy:", accuracy)

        if accuracy > best_acc:

            best_acc = accuracy

            torch.save(model.state_dict(), "../best_model_rgb_grad_fft.pth")

            print("Best model saved!")

    print("Best Accuracy:", best_acc)


if __name__ == "__main__":

    main()