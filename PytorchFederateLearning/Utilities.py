#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import codecs
import pickle
import sys
import copy
from datetime import datetime

import torch
from sklearn import metrics
from matplotlib import pyplot as plt
from termcolor import colored
from torchvision import datasets, transforms
from FederatedLearning4MultiAgentSystems.PytorchFederateLearning.sampling import mnist_iid, mnist_noniid, \
    mnist_noniid_unequal
from FederatedLearning4MultiAgentSystems.PytorchFederateLearning.sampling import cifar_iid, cifar_noniid


class Utilities:

    def __init__(self):
        pass

    def get_dataset(self, args):
        """ Returns train and test datasets and a user group which is a dict where
        the keys are the user index and the values are the corresponding data for
        each of those users.
        """
        train_dataset = ""
        test_dataset = ""
        user_groups = ""

        if args.dataset == 'cifar':
            data_dir = '../../FederatedLearning4MultiAgentSystems/data/cifar/'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = cifar_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = cifar_noniid(train_dataset, args.num_users)

        elif args.dataset == 'mnist' or 'fmnist':
            if args.dataset == 'mnist':
                data_dir = '../../FederatedLearning4MultiAgentSystems/data/mnist/'
            else:
                data_dir = '../data/fmnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = mnist_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                else:
                    # Chose euqal splits for every user
                    user_groups = mnist_noniid(train_dataset, args.num_users)

        return train_dataset, test_dataset, user_groups

    def plot_confusion_matrix(self, labels, pred_labels):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        cm = metrics.confusion_matrix(labels, pred_labels)
        cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(10))
        cm.plot(values_format='d', cmap='Blues', ax=ax)

    def average_weights(self, w):
        """
        Returns the average of the weights.
        """
        w_avg = []
        for local_weights in w:
            unpickled_local_weights = pickle.loads(codecs.decode(local_weights.encode(), "base64"))
            w_avg = copy.deepcopy(unpickled_local_weights[0])
            for key in w_avg.keys():
                for i in range(1, len(unpickled_local_weights)):
                    w_avg[key] += unpickled_local_weights[i][key]
                w_avg[key] = torch.div(w_avg[key], len(unpickled_local_weights))

        return w_avg

    def exp_details(self, args):
        print('\nExperimental details:')
        print(f'    Model     : {args.model}')
        print(f'    Optimizer : {args.optimizer}')
        print(f'    Learning  : {args.lr}')
        print(f'    Global Rounds   : {args.epochs}\n')

        print('    Federated parameters:')
        if args.iid:
            print('    IID')
        else:
            print('    Non-IID')
        print(f'    Fraction of users  : {args.frac}')
        print(f'    Local Batch size   : {args.local_bs}')
        print(f'    Local Epochs       : {args.local_ep}\n')
        return

    def save_outputs(self, data, file_name):
        # datetime object containing current date and time
        print(colored('=' * 30, 'blue'))
        now = datetime.now()
        print(colored('=' * 30, 'blue'))
        print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S").replace('/', '_').replace(':', '_')
        print("date and time =", dt_string)
        print(colored('=' * 30, 'blue'))

        to_save = file_name + "_" + dt_string + "_.txt"

        with open(to_save, 'w') as f:
            for line in data:
                f.write(str(line))
                f.write(',')
            f.write('\n')
            #f.close()
