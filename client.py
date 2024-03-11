from collections import OrderedDict
from typing import Dict, Tuple

from flwr.common import NDArrays

from model import Net, train, test
import torch
import flwr as fl


class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader, 
                 num_classes
                 )->None:
        super().__init__()

        self.trainloader=trainloader
        self.valloader=valloader

        self.model=Net(num_classes) #initialized with random weights

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(f"Device: {self.device} ")


    def set_parameters(self, parameters):
        #it recives the parameters from the server
        params_dict=zip(self.model.state_dict().keys(), parameters)

        state_dict=OrderedDict({k: torch.Tensor(v) for k,v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    
    def get_parameters(self, config):
        #it sends back parameters to the server
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        #parameters is a list of numpy arrays representing the current state of the global model
        #config is a python dictionary with additional information

        # When this client starts the computation, it receives the weights from the server so we want to 
        # overwrite the initial random weights with the ones sent from the server

        self.set_parameters(parameters)

        # Local training of the model

        lr=config['lr']
        momentum=config['momentum']
        epochs=config['local_epochs']

        optim=torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        train(self.model, self.trainloader, optim, epochs, self.device)

        # Send back the updated model paraemters to the server with get_parameters()
        # it also returns the number of samples used by this client for training
        # and metrics about training

        return self.get_parameters({}), len(self.trainloader), {}

   
    def evaluate(self, parameters: NDArrays, config):
        #it receives the global model from the server and evaluate it on the validation set of this client

        self.set_parameters(parameters)
        
        loss, accuracy=test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {'accuracy': accuracy}



def generate_client_fn(trainloaders, valloaders, num_classes):
    """Function to be called in main.py and then it will be passed to the server so that the server 
    can use it to spawn a client with a certain cid"""
    def client_fn(cid:str):

        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_classes=num_classes)

    return client_fn
    
