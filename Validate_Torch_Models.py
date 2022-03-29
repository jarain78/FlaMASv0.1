import os
import torch
import torchvision
import matplotlib.pyplot as plt
from termcolor import colored
import torch.nn.functional as F
from torchviz import make_dot
from FederatedLearning4MultiAgentSystems.PytorchFederateLearning.Federated import Federated
from sklearn.metrics import confusion_matrix
import seaborn as sns

batch_size_test = 1000
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('test_files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------

PATH = 'TorchModels/'
device = torch.device('cpu')

model_file_list = os.listdir(PATH)

federated_learning = Federated()
federated_learning.build_model()
print(colored('=' * 30, 'blue'))
federated_learning.print_model()
print(colored('=' * 30, 'blue'))
federated_learning.set_model()

test_losses = []

for i_model in model_file_list:
    if i_model != 'README':
        federated_learning.global_model.load_state_dict(torch.load(PATH + i_model, map_location=device))
        federated_learning.global_model.eval()

        model_network = federated_learning.global_model

        test_loss = 0
        correct = 0
        CM = 0

        # Initialize the prediction and label lists(tensors)
        predlist = torch.zeros(0, dtype=torch.long, device='cpu')
        lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

        with torch.no_grad():
            for data, target in test_loader:
                output = model_network(data)

                #make_dot(output.mean(), params=dict(model_network.named_parameters()), show_attrs=True, show_saved=True).render("test.png", format="png")
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                classes = target.data.view_as(pred)
                # Append batch prediction results
                predlist = torch.cat([predlist, pred.view(-1).cpu()])
                lbllist = torch.cat([lbllist, classes.view(-1).cpu()])


        print(colored('-' * 30, 'blue'))
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Model Name: \n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)), i_model)

        # Confusion matrix
        conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())

        print(conf_mat)

        # Per-class accuracy
        class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
        print(class_accuracy)

        if "pepita_hp_1" in i_model:
            # Normalise
            cmn = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(cmn* 100, annot=True, fmt='.2f',  cmap='Blues')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show(block=False)

            #plot_confusion_matrix(conf_mat, [0,1,2,3,4,5,6,7,8,9], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
        elif "pepita_hp_2"  in i_model:
            # Normalise
            cmn = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(cmn * 100, annot=True, fmt='.2f', cmap='Blues')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show(block=False)

            #plot_confusion_matrix(conf_mat, [0,1,2,3,4,5,6,7,8,9], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
        elif "pepita_hp_3"  in i_model:
            # Normalise
            cmn = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(cmn * 100, annot=True, fmt='.2f', cmap='Blues')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show(block=False)

            #plot_confusion_matrix(conf_mat, [0,1,2,3,4,5,6,7,8,9], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
