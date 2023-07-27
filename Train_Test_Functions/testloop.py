import torch
import torch.nn as nn

from Data_Pre_processing.Feature_Sample import Feature_Sample


def test_loop(testloader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_func = nn.CrossEntropyLoss()

    size = len(testloader.dataset)
    num_batches = len(testloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x_d = Feature_Sample(x, model.input_size).to(device)
            y_d = y.to(device)
            pred = model(x_d)
            test_loss += loss_func(pred, y_d).item()
            correct += (pred.argmax(1) == y_d).type(torch.float).sum().item()
    test_loss = test_loss / num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.2f}%, "
          f"Avg loss: {test_loss:>8f}")

    return test_loss, accuracy
