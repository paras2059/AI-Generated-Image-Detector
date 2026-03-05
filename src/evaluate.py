import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import ImageDataset
from model import GradCNN


dataset = ImageDataset("../data/test_small")

loader = DataLoader(dataset, batch_size=8)

device = torch.device("cpu")

model = GradCNN().to(device)

model.load_state_dict(torch.load("../best_model_rgb_grad_fft.pth", map_location=device))

model.eval()

all_labels = []
all_probs = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)[:,1]

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())


preds = [1 if p > 0.5 else 0 for p in all_probs]


# ======================
# Confusion Matrix
# ======================

cm = confusion_matrix(all_labels, preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ======================
# ROC Curve
# ======================

fpr, tpr, _ = roc_curve(all_labels, all_probs)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label="AUC = %.3f" % roc_auc)

plt.plot([0,1],[0,1],'--')

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.show()


# ======================
# Precision Recall Curve
# ======================

precision, recall, _ = precision_recall_curve(all_labels, all_probs)

plt.figure()

plt.plot(recall, precision)

plt.title("Precision-Recall Curve")

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.show()