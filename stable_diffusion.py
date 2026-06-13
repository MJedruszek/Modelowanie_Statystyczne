import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm

# Import Hugging Face Diffusers components
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DModel, UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor
from peft import LoraConfig

# 1. Dataset for Local Images
class LocalImageDataset(Dataset):
    def __init__(self, folder_path, tokenizer, size=512):
        self.image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
        self.tokenizer = tokenizer
        self.size = size
        
        # Stable Diffusion expects images normalized between -1 and 1
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image and convert to RGB
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)
        
        # Tokenize our caption condition: "a photo of a cat"
        text_inputs = self.tokenizer(
            "a photo of a cat",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(0)
        }

# 2. The Main Training & Generation Pipeline
def train_stable_diffusion(data_folder, epochs=5, batch_size=2, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    
    # Load all individual components from the pretrained pipeline
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze VAE and Text Encoder weights
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) # Freeze base UNet, we only train LoRA

    # Inject LoRA layers into the UNet attention layers
    lora_config = LoraConfig(
        r=4,                                      # The Rank size
        lora_alpha=4,                             # Scaling factor 
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], # Standard SD 1.5 attention nodes
    )

    # Seamlessly inject the LoRA layers into your UNet
    unet.add_adapter(lora_config)
    
    # Filter optimizer to only watch the freshly added LoRA weights
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(lora_layers, lr=lr)

    # Prepare Data
    dataset = LocalImageDataset(data_folder, tokenizer, size=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("--- Starting Stable Diffusion LoRA Training ---")
    unet.train()
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            # Move data to GPU
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # 1. Convert real cat images into Latents using frozen VAE encoder
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215 # Scale factor used by SD 1.5

            # 2. Sample random noise to add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            # Add noise to the latents according to the forward diffusion process
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3. Get the text embedding conditioning ("a photo of a cat")
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # 4. UNet predicts the noise residue using latent features + time + text context
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # 5. Calculate loss against the actual noise injected
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item()})

    # Save the specialized LoRA weights separately
    os.makedirs("sd_cat_lora", exist_ok=True)
    unet.save_attn_procs("sd_cat_lora")
    print("Training finished! LoRA weights saved to 'sd_cat_lora'.\n")

    # ----------------------------------------------------
    # PHASE 3: Generation (Inference)
    # ----------------------------------------------------
    print("--- Running Pipeline Inference ---")
    # Load a fresh pipeline with standard weights
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    # Load your trained cat weights straight into the pipeline!
    pipeline.unet.load_attn_procs("sd_cat_lora")

    # Generate a brand new cat image using text conditioning
    prompt = "a photo of a cat, high resolution, detailed masterpiece"
    image = pipeline(prompt, num_inference_steps=30).images[0]
    
    image.save("stable_diffusion_cat_10.png")
    print("Generated image successfully saved to 'stable_diffusion_cat.png'!")

    prompt = "a photo of a cat, high resolution"
    image = pipeline(prompt, num_inference_steps=30).images[0]
    
    image.save("stable_diffusion_cat2_10.png")
    print("Generated image successfully saved to 'stable_diffusion_cat2.png'!")

    prompt = "a photo of a cat"
    image = pipeline(prompt, num_inference_steps=30).images[0]
    
    image.save("stable_diffusion_cat3_10.png")
    print("Generated image successfully saved to 'stable_diffusion_cat3.png'!")

if __name__ == "__main__":
    # Ensure this folder path points to your dataset folder containing .jpg/.png images
    train_stable_diffusion(data_folder="PetImages/Cat", epochs=10, batch_size=1)