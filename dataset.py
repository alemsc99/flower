import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


def get_mnist(data_path:str='./data'):

    tr= Compose([ToTensor(), Normalize((0.1307), (0.3081,))])
    trainset=MNIST(data_path, train=True, download=True, transform=tr)
    testset=MNIST(data_path, train=False, download=True, transform=tr)


    return trainset, testset



def prepare_dataset(num_partitions:int, batch_size:int, val_ratio: float=0.1):
    # num_partions= number of clients= number of partitions to create starting from the training set
    # batch_size= assuming it's the same for every client
    # val_ratio= ratio of validation samples wrt training samples
    trainset, testset=get_mnist()
    # We want to split our trainset in #num_partitions portions 
    num_images=len(trainset)//num_partitions #number of images in each partition
   
    partition_len=[num_images]*num_partitions #a list of lenghts of every partition
    trainsets=random_split(trainset, partition_len, torch.Generator().manual_seed(2024))
    # trainsets is going to be a list of datasets each of them with num_images samples

    #Now, for each trainingset, we want to partition it in training and validation 
    #and create DataLoaders for each of them
    trainloaders=[]
    valloaders=[]
    for trainset_ in trainsets:
        num_total=len(trainset_)
        num_val=int(val_ratio*num_total)
        num_train=num_total-num_val

        for_train, for_val=random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2024))
        #Now, we have a training set and a validation set, for every partition

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))


    # We would have one dataloader per client for both training and validation set 
    # assgined to this particular client
    
    # We have to create the dataloader for the test set. We use an higher batch size since we
    # want a faster evaluation phase
    testloader=DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader


    

# import torch
# from torch.utils.data import random_split, DataLoader
# from torchvision.transforms import ToTensor, Normalize, Compose
# from torchvision.datasets import MNIST


# def get_mnist(data_path: str = "./data"):
    

#     tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

#     trainset = MNIST(data_path, train=True, download=True, transform=tr)
#     testset = MNIST(data_path, train=False, download=True, transform=tr)

#     return trainset, testset


# def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
#     """Download MNIST and generate IID partitions."""

#     # download MNIST in case it's not already in the system
#     trainset, testset = get_mnist()

#     # split trainset into `num_partitions` trainsets (one per client)
#     # figure out number of training examples per partition
#     num_images = len(trainset) // num_partitions

#     # a list of partition lenghts (all partitions are of equal size)
#     partition_len = [num_images] * num_partitions

#     # split randomly. This returns a list of trainsets, each with `num_images` training examples
#     # Note this is the simplest way of splitting this dataset. A more realistic (but more challenging) partitioning
#     # would induce heterogeneity in the partitions in the form of for example: each client getting a different
#     # amount of training examples, each client having a different distribution over the labels (maybe even some
#     # clients not having a single training example for certain classes). If you are curious, you can check online
#     # for Dirichlet (LDA) or pathological dataset partitioning in FL. A place to start is: https://arxiv.org/abs/1909.06335
#     trainsets = random_split(
#         trainset, partition_len, torch.Generator().manual_seed(2023)
#     )

#     # create dataloaders with train+val support
#     trainloaders = []
#     valloaders = []
#     # for each train set, let's put aside some training examples for validation
#     for trainset_ in trainsets:
#         num_total = len(trainset_)
#         num_val = int(val_ratio * num_total)
#         num_train = num_total - num_val

#         for_train, for_val = random_split(
#             trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
#         )

#         # construct data loaders and append to their respective list.
#         # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
#         trainloaders.append(
#             DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
#         )
#         valloaders.append(
#             DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
#         )

#     # We leave the test set intact (i.e. we don't partition it)
#     # This test set will be left on the server side and we'll be used to evaluate the
#     # performance of the global model after each round.
#     # Please note that a more realistic setting would instead use a validation set on the server for
#     # this purpose and only use the testset after the final round.
#     # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
#     # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
#     # in main.py above the strategy definition for more details on this)
#     testloader = DataLoader(testset, batch_size=128)

#     return trainloaders, valloaders, testloader