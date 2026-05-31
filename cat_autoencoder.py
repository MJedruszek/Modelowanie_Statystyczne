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
            # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 64 x 8 x 8 
            # nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),  # 8 x 8 x 8
            nn.ReLU()
        )
        
        # decoder
        self.decoder = nn.Sequential(
           nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),  # 64 x 8 x 8
            nn.ReLU(),
            # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 64 x 16 x 16
            # nn.ReLU(),
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
    #v1 - podstawowy
    #v2 - dodana warstwa conv 64,64 jako przedostatnia
    torch.save(model.state_dict(), "cat_autoencoder_v2.pth")
    return model

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=8, time_dim=32):
        super(SimpleUNet, self).__init__()
        
        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # downsampling
        self.inc = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        
        # time embedding into layers
        self.time_proj1 = nn.Linear(time_dim, 64)
        self.time_proj2 = nn.Linear(time_dim, 32)
        
        # upsampling
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.outc = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # t shape: [batch, 1]
        t_emb = self.time_mlp(t) # Shape: [batch, time_dim]
        
        # down
        h1 = torch.relu(self.inc(x)) # [batch, 32, 16, 16]
        h2 = self.down1(h1)          # [batch, 64, 8, 8]
        
        # add time
        t_proj1 = self.time_proj1(t_emb)[:, :, None, None] # [batch, 64, 1, 1]
        h2 = h2 + t_proj1
        
        # up
        h3 = torch.relu(self.up1(h2)) # [batch, 32, 16, 16]
        
        # add time again
        t_proj2 = self.time_proj2(t_emb)[:, :, None, None] # [batch, 32, 16, 16]
        h3 = h3 + t_proj2
        
        return self.outc(h3)
    
def train_diffusion(cat_folder, autoencoder_path, timesteps=200):
    # load autoencoder
    ae = CatAutoencoder()
    ae.load_state_dict(torch.load(autoencoder_path))
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False
        
    # initialize unet and dataset
    unet = SimpleUNet(in_channels=8)
    dataset = CatDataset(cat_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = optim.Adam(unet.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    beta = torch.linspace(0.0001, 0.02, timesteps)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    
    # train
    epochs = 30
    # with torch.no_grad():
    #     sample_batch = next(iter(dataloader))
    #     sample_latents = ae.encoder(sample_batch)
    #     latent_std = sample_latents.std().item()
    #     custom_scale = 1.0 / latent_std
    #     print(f"--- DIAGNOSTIC ---")
    #     print(f"Latent Min: {sample_latents.min().item():.4f}, Max: {sample_latents.max().item():.4f}")
    #     print(f"Latent Std Dev: {latent_std:.4f}")
    #     print(f"YOUR ACTUAL SCALING FACTOR SHOULD BE: {custom_scale:.5f}")
    #     print(f"------------------")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # use autoencoder
            with torch.no_grad():
                clean_latents = ae.encoder(batch) 
                clean_latents = clean_latents * 0.54985
            
            # get a timestep (random) and generate noise
            batch_size = batch.size(0)
            t = torch.randint(0, timesteps, (batch_size, 1), dtype=torch.float32)
            noise = torch.randn_like(clean_latents)
            
            # add time:
            # q_sample math: sqrt(alpha_cumprod)*clean + sqrt(1-alpha_cumprod)*noise
            a_cum = alpha_cumprod[t.long()].view(batch_size, 1, 1, 1)
            noisy_latents = torch.sqrt(a_cum) * clean_latents + torch.sqrt(1 - a_cum) * noise
            
            # predict noise and meaure loss
            predicted_noise = unet(noisy_latents, t / timesteps)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")
        
    torch.save(unet.state_dict(), "latent_unet.pth")
    return unet
    

def generate_new_cat(autoencoder_path, unet_path, timesteps=200):
    # load both models
    ae = CatAutoencoder()
    ae.load_state_dict(torch.load(autoencoder_path))
    ae.eval()
    
    unet = SimpleUNet(in_channels=8)
    unet.load_state_dict(torch.load(unet_path))
    unet.eval()
    
    # set the inverse math up
    beta = torch.linspace(0.0001, 0.02, timesteps)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    
    # 1. start with pure static in the latent space dimensions
    x_t = torch.randn(1, 8, 16, 16) 
    
    SCALE_FACTOR = 0.54985
    # 2. denoise step-by-step backwards
    with torch.no_grad():
        for i in reversed(range(timesteps)):
            t_tensor = torch.tensor([[i]], dtype=torch.float32)
            
            # predict noise
            pred_noise = unet(x_t, t_tensor / timesteps)
            
            # reverse diffusion step math to subtract noise safely
            a = alpha[i]
            a_cum = alpha_cumprod[i]
            beta_i = beta[i]
            
            if i > 0:
                z = torch.randn_like(x_t)
            else:
                z = 0 # No noise added on the very last step
                
            # compute previous step's latent representation
            x_t = (1 / torch.sqrt(a)) * (x_t - ((1 - a) / torch.sqrt(1 - a_cum)) * pred_noise) + torch.sqrt(beta_i) * z
            
        # 3. pass the cleaned up latent feature into autoencoder
        x_t = x_t / SCALE_FACTOR
        generated_image = ae.decoder(x_t)
        
    # convert PyTorch tensor back to an image
    generated_image = generated_image.squeeze(0).permute(1, 2, 0).numpy() * 255.0
    generated_image = cv2.cvtColor(generated_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    cv2.imwrite("generated_cat.jpg", generated_image)
    print("New image saved as generated_cat.jpg")


if __name__ == "__main__":
    #only autoencoder
    # print("Start training...")
    # trained_model = train_autoencoder("PetImages/Cat")
    # print("Training done")

    #Unet with autoencoder
    DATASET_FOLDER = "PetImages/Cat"
    AUTOENCODER_PATH = "cat_autoencoder_v1.pth"
    UNET_PATH = "latent_unet.pth"
    TIMESTEPS = 200
    
    #training encoder
    if not os.path.exists(AUTOENCODER_PATH):
        print("--- Phase 1: Training Autoencoder ---")
        trained_ae = train_autoencoder(DATASET_FOLDER)
        print("Autoencoder training complete and model saved.\n")
    else:
        print(f"--- Phase 1: Found existing Autoencoder weights at '{AUTOENCODER_PATH}' ---")
        print("Skipping Autoencoder training.\n")

    #training unet
    print("--- Phase 2: Training Denoising UNet ---")
    trained_unet = train_diffusion(
        cat_folder=DATASET_FOLDER, 
        autoencoder_path=AUTOENCODER_PATH, 
        timesteps=TIMESTEPS
    )
    print("Diffusion UNet training complete and model saved.\n")

    #testing image generation
    print("--- Phase 3: Generating a Brand New Cat Image ---")
    generate_new_cat(
        autoencoder_path=AUTOENCODER_PATH, 
        unet_path=UNET_PATH, 
        timesteps=TIMESTEPS
    )
    print("Pipeline execution complete successfully!")