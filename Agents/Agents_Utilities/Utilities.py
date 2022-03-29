from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored


class Agents_Utilities:

    def __init__(self):
        pass

    def plot_most_incorrect(self, incorrect, n_images):

        rows = int(np.sqrt(n_images))
        cols = int(np.sqrt(n_images))

        fig = plt.figure(figsize=(20, 10))
        for i in range(rows * cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            image, true_label, probs = incorrect[i]
            true_prob = probs[true_label]
            incorrect_prob, incorrect_label = torch.max(probs, dim=0)
            ax.imshow(image.view(28, 28).cpu().numpy(), cmap='bone')
            ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n' \
                         f'pred label: {incorrect_label} ({incorrect_prob:.3f})')
            ax.axis('off')
        fig.subplots_adjust(hspace=0.5)

    def get_predictions(self, model, iterator, device):

        model.eval()

        images = []
        labels = []
        probs = []

        with torch.no_grad():
            for (x, y) in iterator:
                x = x.to(device)

                y_pred, _ = model(x)

                y_prob = F.softmax(y_pred, dim=-1)
                top_pred = y_prob.argmax(1, keepdim=True)

                images.append(x.cpu())
                labels.append(y.cpu())
                probs.append(y_prob.cpu())

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        probs = torch.cat(probs, dim=0)

        return images, labels, probs

    def load_saved_model(self):
        model_path = '../../TorchModels/FL_2021-11-23 08:41:36.903909AgentName: pepita_hp_3.pt'
        # model.load_state_dict(torch.load(model_path))
        model_l = torch.load(model_path)
        # self.get_predictions(model_l, iterator, device)


if __name__ == '__main__':
    utilities = Agents_Utilities()
    utilities.load_saved_model()
