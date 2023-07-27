import torch
import torch.nn as nn
from Data_Pre_processing.Feature_Sample import Feature_Sample


def train_loop(trainloader, model, local_epochs, learning_rate=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(local_epochs):
        print(f'Epoch{i + 1}/{local_epochs} begin!')

        size = len(trainloader.dataset)
        cum_n = 0

        print(f"Model[{cum_n:>5d}/{size:>5d}]")
        for batch, (x, y) in enumerate(trainloader):
            cum_n += len(x)
            x_d = Feature_Sample(x, model.input_size).to(device)
            y_d = y.to(device)
            pred = model(x_d)
            loss = loss_func(pred, y_d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ((batch + 1) % 10 == 0) or (cum_n == size):
                print(f"Model[{cum_n:>5d}/{size:>5d}]")

        print(f'Epoch{i + 1}/{local_epochs} completed!')
