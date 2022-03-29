import os
import copy
import time
import pickle
import numpy as np
from termcolor import colored
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
from FederatedLearning4MultiAgentSystems.PytorchFederateLearning.options import args_parser
from FederatedLearning4MultiAgentSystems.PytorchFederateLearning.update import LocalUpdate
from FederatedLearning4MultiAgentSystems.PytorchFederateLearning.models import MLP, CNNMnist, CNNFashion_Mnist, \
    CNNCifar, CNNCustom
from FederatedLearning4MultiAgentSystems.PytorchFederateLearning.Utilities import Utilities


class Federated:

    def __init__(self):

        # ------------------------------------------------------------------------------------------------------------------
        # Training
        self.train_loss, self.train_accuracy = [], []
        self.val_acc_list, self.net_list = [], []
        self.cv_loss, self.cv_acc = [], []
        self.print_every = 2
        self.val_loss_pre, self.counter = 0, 0

        self.start_time = time.time()
        self.best_valid_loss = 0
        # ------------------------------------------------------------------------------------------------------------------
        # define paths
        self.path_project = os.path.abspath('../..')
        self.logger = SummaryWriter('../logs')

        self.args = args_parser()
        print(colored('=' * 30, 'green'))
        print(self.args)
        print(colored('=' * 30, 'green'))

        # exp_details(self.args)

        self.utilities = Utilities()
        # Define the different parameters to configure the training
        if self.args.gpu:
            torch.cuda.set_device(self.args.gpu)
        self.device = 'cuda' if self.args.gpu else 'cpu'

        # load dataset and user groups
        self.train_dataset, self.test_dataset, self.user_groups = self.utilities.get_dataset(self.args)

        self.save_tr_acc = []
        self.save_tr_loss = []

        self.save_te_acc = []
        self.save_tea_acc = []

    def build_model(self):
        # BUILD MODEL
        if self.args.model == 'cnn':
            # Convolutional neural netork
            if self.args.dataset == 'mnist':
                self.global_model = CNNMnist(args=self.args)
            elif self.args.dataset == 'fmnist':
                self.global_model = CNNFashion_Mnist(args=self.args)
            elif self.args.dataset == 'cifar':
                self.global_model = CNNCifar(args=self.args)

        elif self.args.model == 'mlp':
            # Multi-layer preceptron
            img_size = self.train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                self.global_model = MLP(dim_in=len_in, dim_hidden=64,
                                        dim_out=self.args.num_classes)
        elif self.args.dataset == 'custom':
            self.global_model = CNNCustom(args=self.args)

        else:
            exit('Error: unrecognized model')

    def set_model(self):
        # ------------------------------------------------------------------------------------------------------------------
        # Set the model to train and send it to device.
        self.global_model.to(self.device)
        # ------------------------------------------------------------------------------------------------------------------
        # copy weights
        self.global_weights = self.global_model.state_dict()

    def print_model(self):
        print(self.global_model)

    def train_global_model(self):
        # Train global model
        self.global_model.train()

    def train_local_model(self, AgName="Nada", epoch=1):
        self.start_time = time.monotonic()
        local_weights, local_losses = [], []

        # print(f'\n | Global Training Round : {epoch + 1} |\n')

        local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                  idxs=self.user_groups[0], logger=self.logger)

        w, loss = local_model.update_weights(
            model=copy.deepcopy(self.global_model), global_round=epoch)

        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))
        self.get_acc(local_model, AgName=AgName, n_user=1, epoch=1)

        self.end_time = time.monotonic()

        return local_weights, local_losses

    def get_acc(self, local_model, AgName="Nada", n_user=1, epoch=1):
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        self.global_model.eval()

        for c in range(n_user):
            acc, loss = local_model.inference(model=self.global_model)
            list_acc.append(acc)
            list_loss.append(loss)

        self.train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        print(colored('=' * 30, 'green'))
        if (epoch + 1) % self.print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(self.train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * self.train_accuracy[-1]))

        # Test inference after completion of training
        test_acc, test_loss = local_model.test_inference(self.args, self.global_model, self.test_dataset)

        print(f' \n Results after {self.args.epochs} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100 * self.train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
        print(colored('=' * 30, 'green'))

        # Train-Info
        self.save_tr_acc.append(100 * self.train_accuracy[-1])
        self.save_tr_loss.append(np.mean(np.array(self.train_loss)))

        self.save_te_acc.append(100 * test_acc)
        self.save_tea_acc.append(100 * self.train_accuracy[-1])

        print(self.save_tr_acc)
        print("="*20)
        print(self.save_tr_loss)
        print("="*20)

        print(self.save_te_acc)
        print("="*20)
        print(self.save_tea_acc)
        print("="*20)


        if test_loss < self.best_valid_loss:
            self.best_valid_loss = test_loss
            dateTimeObj = datetime.now()

            torch.save(self.global_model.state_dict(),
                       '../TorchModels/FL_' + str(dateTimeObj) + "AgentName: " + AgName + '.pt')

            '''file_name = '/home/jarain78/Pycharm_Projects/Manhattan_Project/FederatedLearning4MultiAgentSystems/Agents/Outputs/Train_' + AgName
            train_data = str(np.mean(np.array(self.train_loss))) + "," + str(100 * self.train_accuracy[-1])
            self.utilities.save_outputs(train_data, file_name)

            # Test-Info
            file_name = '/home/jarain78/Pycharm_Projects/Manhattan_Project/FederatedLearning4MultiAgentSystems/Agents/Outputs/Test_' + AgName
            train_data = [100 * self.train_accuracy[-1], 100 * test_acc]
            self.utilities.save_outputs(train_data, file_name)'''

            '''self.save_tr_acc = []
            self.save_tr_loss = []

            self.save_te_acc = []
            self.save_tea_acc = []'''

    def print_all(self):
        print(self.save_tr_acc)
        print(self.save_tr_loss)

        print(self.save_te_acc)
        print(self.save_tea_acc)


    def average_all_weights(self, local_weights, local_losses, verbose=False):

        # update global weights
        global_weights = self.utilities.average_weights(local_weights)

        # update global weights
        self.global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        self.train_loss.append(loss_avg)

    def add_new_local_weight_local_losses(self, local_weights, local_losses):
        # update global weights
        self.global_model.load_state_dict(local_weights)
        self.train_loss.append(local_losses)

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
