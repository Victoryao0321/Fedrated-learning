import random
import pandas as pd
import torch
from Train_Test_Functions.trainloop import train_loop
from Train_Test_Functions.testloop import test_loop
from Data_Pre_processing.Load_Testset import testloader


def Fed_Ensemble_Train(trainloaders: list,
                       model_list: list,
                       model_name_list: list,
                       window_size: tuple,
                       local_epochs: int,
                       num_communication_loops: int):
    '''
    trainloaders: 包含每个客户端的dataloader列表
    model_list: 一个双重列表，有几个子列表代表有几种异构模型，子列表长度代表每种模型的数量，一般是n*10
    model_name_list: 模型名字列表，用于输出训练日志和保存最终模型参数
    window_size: 一个元组，(min_window_size,max_window_size)，一般是5*2
    local_epochs: 客户端本地训练轮数
    num_communication_loops: 通信轮数，等于采样比例*客户端数量
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def create_window_list():
        # 用于生成窗口大小列表，一般为[16, 16, 17, 17, 18, 18, 19, 19, 20, 20]
        basic_list = list(range(window_size[0], window_size[-1] + 1, 1))
        times = (len(model_list[0]) // ((window_size[-1] - window_size[0]) + 1)) + 1
        window_list = (basic_list * times)[:len(model_list[0])]
        window_list = sorted(window_list)
        return window_list

    def Create_Sampled_Client_list():
        # 生成每个模型，每轮通信抽取的客户端列表
        Sampled_Client_list = [[] for _ in range(num_communication_loops)]
        for Client_list in Sampled_Client_list:
            for _ in range(len(model_list)):
                Client_list.append(random.sample(range(10), 10))
        return Sampled_Client_list

    window_list = create_window_list()
    Sampled_Client_list = Create_Sampled_Client_list()  # 一个列表里有n个通信轮，每轮里3个模型客户端列表，每个客户端列表里10个

    # 给每个模型添加窗口大小属性
    for sub_model_list in model_list:
        for sub_model, size in zip(sub_model_list, window_list):
            sub_model.input_size = size

    # 开始通信
    df_Result = pd.DataFrame()
    for communication_loop, each_communication_loop_Client_ID_list in enumerate(Sampled_Client_list, 1):
        print('-' * 100)
        print(f'Communication_loop {communication_loop} Begin!')

        for sub_model_list, model_name, Model_Client_list in zip(model_list, model_name_list,
                                                                 each_communication_loop_Client_ID_list):
            print(f'{model_name} Begin!')

            for num, (sub_model, Client_ID) in enumerate(zip(sub_model_list, Model_Client_list), 1):
                print(f'{model_name} {num} Begin!')
                train_loop(trainloader=trainloaders[Client_ID], model=sub_model.to(device), local_epochs=local_epochs)
                print(f'{model_name} {num} Completed!')

            print(f'{model_name} Completed!')
        print(f'Communication_loop {communication_loop} Completed!')

        for sub_model_list, model_name in zip(model_list, model_name_list):
            for model_num, sub_model in enumerate(sub_model_list):
                print(f'Communication_loop_{communication_loop}_{model_name}_{model_num} Test:')
                test_loss, accuracy = test_loop(testloader=testloader, model=sub_model)
                df_Result = df_Result.append(pd.Series([
                    f'Communication_loop_{communication_loop}_{model_name}_{model_num}', test_loss, accuracy]),
                    ignore_index=True)
                torch.save(sub_model.state_dict(),
                           f'Communication_loop_{communication_loop}_{model_name}_{model_num}.pkl')

    df_Result.columns = ['Model_Info', 'Test_Loss', 'Test_Acc']
    df_Result.to_excel('Result.xlsx', index=False, header=True)
    print('Finish!')
