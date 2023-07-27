import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_testset():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    testset = torchvision.datasets.MNIST(root=r'Raw_data',
                                         train=False,
                                         transform=transform,
                                         download=False)
    testloader = DataLoader(testset, batch_size=len(testset), shuffle=False)
    return testloader


testloader = load_testset()
