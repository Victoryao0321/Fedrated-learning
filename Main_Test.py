import pandas as pd
from Data_Pre_processing.Load_Testset import testloader
from Data_Pre_processing.Feature_Sample import Feature_Sample
from Models.Load_Model import Load_Model
from Models.Basic_Model1 import DenseNet
from Models.Basic_Model2 import ResNet
from Models.Basic_Model3 import EfficientNet
import torch
import warnings

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
window_size = sorted(list(range(18, 22 + 1, 1)) * 2)
DenseNet_list = [Load_Model(f'0706_1822_5_10_5/Communication_loop_5_DenseNet_{num}.pkl', DenseNet(1, 10, size)) for
                 num, size in zip(range(10), window_size)]
ResNet_list = [Load_Model(f'0706_1822_5_10_5/Communication_loop_5_ResNet_{num}.pkl', ResNet(1, 10, size)) for num, size
               in zip(range(10), window_size)]
EfficientNet_list = [
    Load_Model(f'0706_1822_5_10_5/Communication_loop_5_EfficientNet_{num}.pkl', EfficientNet(1, 10, size)) for
    num, size in zip(range(10), window_size)]
Full_Model_list = [*DenseNet_list, *ResNet_list, *EfficientNet_list]


def Four_Ensemble_Strategies(testloader, Full_Model_list, times_for_each_model, num_winners):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Ensemble_Result = pd.DataFrame()
    Predicted_data = []
    with torch.no_grad():
        for n, model in enumerate(Full_Model_list):
            model = model.to(device)
            for t in range(times_for_each_model):
                for x, y in testloader:
                    x_d = Feature_Sample(x, model.input_size).to(device)
                    pred = model(x_d).cpu().numpy()
                    Predicted_data.append(pd.DataFrame(pred))
                    print(f'model{n} predict completed!')

    def Unweighted_average(df):
        return df.mean().argmax()

    def Majority_vote(df):
        return pd.value_counts(df.idxmax(axis=1)).index[0]

    def get_winners(df):
        return df.iloc[df.max(axis=1).sort_values(ascending=False).index[:num_winners], :]

    for num, sampleID in enumerate(range(Predicted_data[0].shape[0])):
        temp_result = pd.concat([df.iloc[[sampleID]] for df in Predicted_data], ignore_index=True)
        Ensemble_Result = Ensemble_Result.append(pd.Series([
            Unweighted_average(temp_result),
            Majority_vote(temp_result),
            Unweighted_average(get_winners(temp_result)),
            Majority_vote(get_winners(temp_result)),
        ]), ignore_index=True)

        if num % 10 == 0:
            print(f'{num} completed!')

    label = pd.Series([y.item() for _, y in testloader])
    Ensemble_Result = pd.concat([Ensemble_Result, label], axis=1)

    return Ensemble_Result


Ensemble_Result = Four_Ensemble_Strategies(testloader=testloader,
                                           Full_Model_list=Full_Model_list,
                                           times_for_each_model=3,
                                           num_winners=45)




