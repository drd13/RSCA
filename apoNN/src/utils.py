import torch
import torchvision



def generate_loss_with_masking(loss):
    def loss_with_masking(x_pred,x_true):
        non_zero = x_true!=0
        return loss(x_pred[non_zero],x_true[non_zero])
    return loss_with_masking
