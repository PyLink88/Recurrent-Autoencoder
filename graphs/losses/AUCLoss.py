import torch
import torch.nn as nn

def AUC_approx(x, x_hat, y, lambda_auc):

    # Computing error for each row
    err = torch.abs(x - x_hat).mean(axis = (1, 2))

    # Selecting error of positive and negative example
    err_n = err[y == 1]
    err_a = err[y > 1]
    n_a = (err_a.shape)[0]
    n_n = (err_n.shape)[0]

    # If there are positive examples compute the AUC penalty
    if n_a > 0:
        diff = err_a.view(-1, 1).unsqueeze(1) - err_n.view(-1, 1)
        exp = torch.sigmoid(diff).sum()
        auc = lambda_auc * exp / (n_a * n_n)
        loss = err.mean() + auc

        return loss

    else:
        loss = err.mean()

        return loss

class AUCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = AUC_approx

    def forward(self, x_hat, x_true, y, lambda_auc):
        loss = self.loss(x_hat, x_true, y, lambda_auc)
        return loss


if __name__ == '__main__':

    #     # Lambda_reg in {0,0.1,1,10,100,1000,10000} con MSE error

    x = torch.rand([10,2,3])
    x_hat =  torch.rand([10,2,3]) +.2
    y = torch.tensor([0,0,0,0,0,0,1,1,1,1])

    loss = AUCLossNN()
    print(loss(x, x_hat, y, 10))
    print(AUCLoss(x, x_hat, y, 10))




