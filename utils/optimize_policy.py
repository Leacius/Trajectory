import torch.nn as nn
# from model import MLP

def get_optim_policies(model, lr=1e-3):
    conv1d = []
    conv1d_bias = []
    linear = []
    linear_bias = []
    classifier = []
    classifier_bias = []
    bn = []
    bn_bias = []
    
    for name, module in model.named_modules():
        ps = list(module.parameters())
        if isinstance(module, nn.Conv1d):
            conv1d.append(ps[0])
            conv1d_bias.append(ps[1])
        elif isinstance(module, nn.Linear):
            linear.append(ps[0])
            linear_bias.append(ps[1])
        elif isinstance(module, nn.Linear) and name == "fc":
            classifier.append(ps[0])
            classifier_bias.append(ps[1])
        elif isinstance(module, nn.BatchNorm1d):
            bn.append(ps[0])
            bn_bias.append(ps[1])
    
    return [
        {"params": conv1d, 'lr_mult': 0.25, 'decay_mult': 1},
        {"params": conv1d_bias, 'lr_mult': 0.5, 'decay_mult': 0},
        {"params": linear, 'lr_mult': 1, 'decay_mult': 1},
        {"params": linear_bias, 'lr_mult': 2, 'decay_mult': 0},
        {"params": classifier, 'lr_mult': 1, 'decay_mult': 1},
        {"params": classifier_bias, 'lr_mult': 2, 'decay_mult': 0},
        {"params": bn, 'lr_mult': 0.5, 'decay_mult': 1},
        {"params": bn_bias, 'lr_mult': 0.5, 'decay_mult': 0},
    ]


# if __name__ == "__main__":
#     model = MLP(10)
#     # get_optim_policies(model)
#     print(list(model.parameters()))