from Data_Pre_processing.MNIST_Split import dataloaders
from Train_Test_Functions.Fed_Ensemble_Train import Fed_Ensemble_Train
from Models.Basic_Model1 import DenseNet
from Models.Basic_Model2 import ResNet
from Models.Basic_Model3 import EfficientNet
from Data_Pre_processing.Load_Testset import testloader
from Train_Test_Functions.testloop import test_loop

DenseNet_list = [DenseNet(in_channels=1, num_classes=10) for _ in range(10)]
ResNet_list = [ResNet(in_channels=1, num_classes=10) for _ in range(10)]
EfficientNet_list = [EfficientNet(in_channels=1, num_classes=10) for _ in range(10)]
Model_list = [DenseNet_list, ResNet_list, EfficientNet_list]
Model_name_list = ['DenseNet', 'ResNet', 'EfficientNet']

if __name__ == '__main__':
    Fed_Ensemble_Train(trainloaders=dataloaders,
                       model_list=Model_list,
                       model_name_list=Model_name_list,
                       window_size=(16, 20),
                       local_epochs=5,
                       num_communication_loops=5)
