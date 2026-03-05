import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


def compute_gradients(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad = cv2.sqrt(gx * gx + gy * gy)

    grad = grad / (grad.max() + 1e-6)

    return grad


def compute_fft(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)

    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)

    magnitude = magnitude / magnitude.max()

    return magnitude


class ImageDataset(Dataset):

    def __init__(self, root_dir):

        self.samples = []

        for label, folder in enumerate(["real", "fake"]):

            folder_path = os.path.join(root_dir, folder)

            for img in os.listdir(folder_path):

                img_path = os.path.join(folder_path, img)

                if img_path.lower().endswith((".png", ".jpg", ".jpeg")):

                    self.samples.append((img_path, label))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        img = cv2.imread(img_path)

        img = cv2.resize(img, (224,224))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        grad = compute_gradients(img)

        fft = compute_fft(img)

        rgb_tensor = transforms.ToTensor()(rgb)

        grad_tensor = torch.tensor(grad).unsqueeze(0)

        fft_tensor = torch.tensor(fft).unsqueeze(0)

        combined = torch.cat([rgb_tensor, grad_tensor, fft_tensor], dim=0)

        return combined.float(), torch.tensor(label)