import torch


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        _temp = torch.nn.functional.log_softmax(outputs, dim=1)
        return self.loss(_temp, targets.argmax(dim=1).long())
