import torch
import torch.nn as nn


def Load_Model(filename, Basic_Model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Basic_Model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    return Basic_Model
