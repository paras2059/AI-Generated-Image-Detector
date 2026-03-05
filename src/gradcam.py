import torch
import cv2
import matplotlib.pyplot as plt
import os
import random

from torchcam.methods import GradCAM

from model import GradCNN
from dataset import compute_gradients, compute_fft


device = torch.device("cpu")

model = GradCNN().to(device)

model.load_state_dict(torch.load("../best_model_rgb_grad_fft.pth", map_location=device))

model.eval()

cam_extractor = GradCAM(model, target_layer="model.layer4")


def process_image(img_path):

    img = cv2.imread(img_path)

    img = cv2.resize(img,(224,224))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    grad = compute_gradients(img)

    fft = compute_fft(img)

    rgb_tensor = torch.tensor(rgb).permute(2,0,1).float()/255

    grad_tensor = torch.tensor(grad).unsqueeze(0).float()

    fft_tensor = torch.tensor(fft).unsqueeze(0).float()

    input_tensor = torch.cat([rgb_tensor, grad_tensor, fft_tensor], dim=0)

    input_tensor = input_tensor.unsqueeze(0).to(device)

    output = model(input_tensor)

    pred_class = output.argmax(dim=1).item()

    activation_map = cam_extractor(pred_class, output)

    heatmap = activation_map[0].squeeze().cpu().numpy()

    return rgb, heatmap


# pick random real image
real_folder = "../data/test_small/real"
real_img = os.path.join(real_folder, random.choice(os.listdir(real_folder)))

# pick random fake image
fake_folder = "../data/test_small/fake"
fake_img = os.path.join(fake_folder, random.choice(os.listdir(fake_folder)))

print("Real image:", real_img)
print("Fake image:", fake_img)


real_rgb, real_heat = process_image(real_img)
fake_rgb, fake_heat = process_image(fake_img)


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(real_rgb)
plt.imshow(real_heat, cmap="jet", alpha=0.5)
plt.title("REAL Image GradCAM")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(fake_rgb)
plt.imshow(fake_heat, cmap="jet", alpha=0.5)
plt.title("FAKE Image GradCAM")
plt.axis("off")

plt.show()