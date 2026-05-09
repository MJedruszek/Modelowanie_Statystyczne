import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob

class CatAutoencoder(nn.Module):
    def __init__(self):
        super(CatAutoencoder, self).__init__()
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 16 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 16 x 16
            nn.ReLU(),
            # 8 x 16 x 16
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),  
            nn.ReLU()
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 16 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 3 x 128 x 128
            nn.Sigmoid() # robi wartosc od 0 do 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    

# data loader
class CatDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = glob.glob(folder_path + "/*.jpg")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # zaladuj obraz
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        
        # corupted obraz
        while img is None:
            idx = (idx + 1) % len(self.image_paths)
            img = cv2.imread(self.image_paths[idx])
            
        # 3. zamien rozmiar i kolejnosc koloru
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return tensor_img

def train_autoencoder(cat_folder):
    model = CatAutoencoder()
    dataset = CatDataset(cat_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = criterion(outputs, batch)
            
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
    # zachowaj model
    torch.save(model.state_dict(), "cat_autoencoder_v1.pth")
    return model

if __name__ == "__main__":
    print("Start training...")
    trained_model = train_autoencoder("PetImages/Cat")
    print("Training done")