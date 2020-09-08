import torch
import torchvision

def generate_loss_with_masking(loss):
    def loss_with_masking(x_pred,x_true,mask):
        return loss(x_pred[mask],x_true[mask])
    return loss_with_masking
    
