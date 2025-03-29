import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets
import torchvision.transforms as transforms




def MNIST_DataLoading(saving_location, image_size, batch_size, data_label=None):


    # loading the dataset
    train_dataset = datasets.MNIST(root=saving_location, download=True, train=True,
                             transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                            ]))
    
    test_dataset = datasets.MNIST(root=saving_location, train=False,
                              transform=transforms.Compose([
                                  transforms.Resize(image_size),
                                  transforms.ToTensor()
                            ]))
    


    if data_label != None:
 
        for times, label_index in enumerate(range(len(data_label))):
            if times == 0:
                train_indices = (train_dataset.targets == data_label[label_index])
                test_indices = (test_dataset.targets == data_label[label_index])
            else:
                train_indices = train_indices | (train_dataset.targets == data_label[label_index])
                test_indices = test_indices | (test_dataset.targets == data_label[label_index])
    

    train_dataset.data, train_dataset.targets = train_dataset.data[train_indices], train_dataset.targets[train_indices]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               drop_last = True,
                                               shuffle=True)
    
    test_dataset.data, test_dataset.targets = test_dataset.data[test_indices], test_dataset.targets[test_indices]

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              drop_last = True,
                                              shuffle=False)
    
    return train_loader, test_loader , train_dataset, test_dataset




def FashionMNIST_DataLoading(saving_location, image_size, batch_size, data_label=None):


    # loading the dataset
    train_dataset = datasets.FashionMNIST(root=saving_location, download=True, train=True, 
                             transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                            ]))
    
    test_dataset = datasets.FashionMNIST(root=saving_location, train=False,
                              transform=transforms.Compose([
                                  transforms.Resize(image_size),
                                  transforms.ToTensor()
                            ]))
    


    if data_label != None:
 
        for times, label_index in enumerate(range(len(data_label))):
            if times == 0:
                train_indices = (train_dataset.targets == data_label[label_index])
                test_indices = (test_dataset.targets == data_label[label_index])
            else:
                train_indices = train_indices | (train_dataset.targets == data_label[label_index])
                test_indices = test_indices | (test_dataset.targets == data_label[label_index])
    

    train_dataset.data, train_dataset.targets = train_dataset.data[train_indices], train_dataset.targets[train_indices]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               drop_last = True,
                                               shuffle=True)
    
    test_dataset.data, test_dataset.targets = test_dataset.data[test_indices], test_dataset.targets[test_indices]

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              drop_last = True,
                                              shuffle=False)
    
    return train_loader, test_loader , train_dataset, test_dataset



def EMNIST_DataLoading(saving_location, image_size, batch_size, data_label=None):


    # loading the dataset
    train_dataset = datasets.EMNIST(root=saving_location, download=True, train=True, split="byclass",
                             transform=transforms.Compose([
                                transforms.Resize(image_size),
                                lambda img: transforms.functional.rotate(img, -90),
                                lambda img: transforms.functional.hflip(img),
                                transforms.ToTensor(),
                            ]))
    
    test_dataset = datasets.EMNIST(root=saving_location, train=False, split="byclass",
                              transform=transforms.Compose([
                                transforms.Resize(image_size),
                                lambda img: transforms.functional.rotate(img, -90),
                                lambda img: transforms.functional.hflip(img),
                                transforms.ToTensor(),
                            ]))
    


    if data_label != None:
 
        for times, label_index in enumerate(range(len(data_label))):
            if times == 0:
                train_indices = (train_dataset.targets == data_label[label_index])
                test_indices = (test_dataset.targets == data_label[label_index])
            else:
                train_indices = train_indices | (train_dataset.targets == data_label[label_index])
                test_indices = test_indices | (test_dataset.targets == data_label[label_index])
    

    train_dataset.data, train_dataset.targets = train_dataset.data[train_indices], train_dataset.targets[train_indices]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               drop_last = True,
                                               shuffle=True)
    
    test_dataset.data, test_dataset.targets = test_dataset.data[test_indices], test_dataset.targets[test_indices]

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              drop_last = True,
                                              shuffle=False)
    
    return train_loader, test_loader , train_dataset, test_dataset