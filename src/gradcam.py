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

    # normalize heatmap
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # convert heatmap to color
    heatmap_color = cv2.applyColorMap((heatmap*255).astype("uint8"), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # overlay heatmap on original image
    overlay = cv2.addWeighted(rgb, 0.6, heatmap_color, 0.4, 0)

    return rgb, overlay


# random real image
real_folder = "../data/test_small/real"
real_img = os.path.join(real_folder, random.choice(os.listdir(real_folder)))

# random fake image
fake_folder = "../data/test_small/fake"
fake_img = os.path.join(fake_folder, random.choice(os.listdir(fake_folder)))

print("Real image:", real_img)
print("Fake image:", fake_img)

real_rgb, real_overlay = process_image(real_img)
fake_rgb, fake_overlay = process_image(fake_img)


plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(real_rgb)
plt.title("Real Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(real_overlay)
plt.title("Real Image + Grad-CAM")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(fake_rgb)
plt.title("Fake Image")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(fake_overlay)
plt.title("Fake Image + Grad-CAM")
plt.axis("off")

plt.tight_layout()

# save figure for README
plt.savefig("../gradcam_comparison.png", dpi=300)

plt.show()
