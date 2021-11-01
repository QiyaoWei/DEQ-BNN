import torchvision
from torchvision.transforms import transforms

class THUCIFAR10(torchvision.datasets.CIFAR10):
    url = "https://cloud.tsinghua.edu.cn/f/5c39f4c5a1224a29b6b6/?dl=1"
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

def get_transform(dataset):
    if(dataset == 'cifar10'):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError(f"{dataset} not implemented")
    return transform

def get_dataset(dataset, download=False):
    transform = get_transform(dataset)
    if(dataset == 'cifar10'):
        trainset = THUCIFAR10(root='./data', train=True, download=download, transform=transform)
        testset = THUCIFAR10(root='./data', train=False, download=False, transform=transform)
        num_classes = 10
        inputs = 3
    else:
        raise NotImplementedError(f"{dataset} not implemented")
    return trainset, testset, inputs, num_classes