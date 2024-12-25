# train.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import os
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFilter
import random
from torchvision import transforms
import torch.nn.functional as F

#device = "cuda"
# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from dataset import create_wall_dataloader
from models import JEPA_Model
from schedulers import Scheduler, LRSchedule


def vicreg_loss(x, y, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):

    repr_loss = F.mse_loss(x, y)

    # Variance
    std_x = torch.sqrt(x.var(dim=0) + 1e-4)
    std_y = torch.sqrt(y.var(dim=0) + 1e-4)
    std_loss = (F.relu(1 - std_x).mean() + F.relu(1 - std_y).mean()) / 2

    # Covariance
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)

    cov_x = (x_centered.T @ x_centered) / (x.size(0)-1)
    cov_y = (y_centered.T @ y_centered) / (y.size(0)-1)

    cov_loss_x = off_diagonal(cov_x).pow(2).sum() / x.size(1)
    cov_loss_y = off_diagonal(cov_y).pow(2).sum() / y.size(1)
    cov_loss = (cov_loss_x + cov_loss_y) / 2

    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    print(f"repr_loss {repr_loss},std_loss {std_loss},cov_loss {cov_loss}")
    return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()


def update_average(beta, old, new):
        if old is None:
            return new
        return old * beta + (1 - beta) * new

def update_moving_average(ma_model, current_model):
    beta = 0.99
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(beta,old_weight, up_weight)

def main():
    parser = argparse.ArgumentParser(description='Train JEPA model')
    parser.add_argument('--data_path', type=str, default='"/scratch/DL24FA/train', help='Path to training data')
    parser.add_argument('--save_path', type=str, default='./model.pth', help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=9, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')

    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight decay')
    parser.add_argument('--learning_rate_weights', default=0.2, type=float, help='Base learning rate for weights')
    parser.add_argument('--learning_rate_biases', default=0.0048, type=float, help='Base learning rate for biases and batch norm parameters')

    args = parser.parse_args()

    print(f"Using device: {device}")
    

    train_loader = create_wall_dataloader(
        data_path=args.data_path,
        probing=False,
        device=device, 
        batch_size=args.batch_size,
        train=True,
    )
    

    model = JEPA_Model()

    model=model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

    import copy
    target_encoder = copy.deepcopy(model.encoder)


    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            
            states = batch.states  
            actions = batch.actions 
            
            B = states.size(0)
            T = actions.size(1) + 1  
            

            states = states.to(device)
            actions = actions.to(device)

            B, T = states.size(0), states.size(1)

            targets = []
            for n in range(T):
                if n==0:
                    s_n_target = model.encoder(states[:, n])
                else:
                    s_n_target = target_encoder(states[:, n])  # (B,64,4,4)
                targets.append(s_n_target)
            s_n_target_all = torch.stack(targets, dim=1) # (B,T,64,4,4)

            s_n = s_n_target_all[:,0]  

            preds = [s_n]
            for n in range(1,T):
                
                s_n = model.predictor(s_n, actions[:, n-1]) # (B,64,4,4)
                preds.append(s_n)
            s_n_pred_all = torch.stack(preds, dim=1)

            pred_encs,s_n_tgt = s_n_pred_all, s_n_target_all

            pred_flat = pred_encs.view(B*T, -1)     # (B*T, C*H*W)
            tgt_flat = s_n_tgt.view(B*T, -1)        # (B*T, C*H*W)
            
            pred_z = pred_flat
            tgt_z = tgt_flat     # (B*T, d)

            loss = vicreg_loss(pred_z, tgt_z, 25, 50, 1)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                update_moving_average(target_encoder,model.encoder)
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
        

    
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()
