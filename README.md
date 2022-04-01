# FLaMAS: Federated Learning based on a SPADE MAS


The implementation presented in this repository is based on the approach presented in the following repository: [Federated-Learning (PyTorch)](https://github.com/AshwinRJ/Federated-Learning-PyTorch).

In this version of federated learning, the agents have been programmed using the [SPADE](https://github.com/javipalanca/spade) tool. This allowed a real distribution to be made, where each of the agents was launched on a different machine. This allowed us to see and analyse the transmission times, as well as the training process of each of the agents.

The experiments were carried out using the MNIST database.  Since the aim of these experiments is to illustrate the effectiveness of the federated learning paradigm, only simple models such as the CNN are used.

## Requirments
Install all the packages from requirments.txt
* Python 3.8.10
* Pytorch 1.9.0+cu102
* Torchvision 0.10.0+cu102
* SPADE 3.2.0
* XMPP server we recommend using [Prosody](https://prosody.im/)

