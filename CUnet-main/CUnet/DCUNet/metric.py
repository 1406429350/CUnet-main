import torch


class STFT_metric:
    def __init__(self):
        self.loss = STFT_

    def __call__(self, outputs1, outputs2, bd):
        loss = self.loss(outputs1, outputs2, bd)
        return loss

def STFT_(outputs1, outputs2, bd):

    output1 = torch.squeeze(outputs1[:, :, :253, :129, :])
    output2 = torch.squeeze(outputs2[:, :, :253, :129, :])
    label1 = bd['y'][:, :253, :129, :]
    label2 = bd['z'][:, :253, :129, :]
    loss1 = torch.mean(torch.abs(output1 - label1))
    loss2 = torch.mean(torch.abs(output2 - label2))
    loss = loss1 + loss2

    return loss.mean()