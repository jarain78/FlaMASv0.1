# Federated-Learning using PyTorch and SPADE


The implementation presented in this repository is based on the approach presented in the following repository: [Federated-Learning (PyTorch)](https://github.com/AshwinRJ/Federated-Learning-PyTorch).

In this version of federated learning, agents have been simulated using the [SPADE](https://github.com/javipalanca/spade) tool. This allowed to perform a real distribution, where each of the agents was launched on a different machine. This allowed the transmission times and training times of each of the agents to be seen and analysed.

The experiments were conducted using the MNIST database.  Since the aim of these experiments is to illustrate the effectiveness of the federated learning paradigm, only simple models such as the CNN are used.


## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision
* SPADE
* XMPP server we recommend using [Prosody](https://prosody.im/)

