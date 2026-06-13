import torch
from diffusers import StableDiffusionPipeline

# 1. Ścieżka do Twoich pobranych wag LoRA
LORA_PATH = "./wagi_sd"

# 2. Ładowanie bazowego modelu Stable Diffusion 1.5 w precyzji float32 (wersja dla CPU)
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
print("Ładowanie modelu bazowego z Hugging Face... (to może chwilę potrwać za pierwszym razem)")

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Wymuszamy uruchomienie na procesorze (CPU)
pipeline = pipeline.to("cpu")

# Optymalizacja pamięci RAM dla słabszych komputerów
pipeline.enable_attention_slicing()

# 3. Wstrzykiwanie Twoich lokalnych wag kota do modelu
print("Ładowanie wag kota...")
pipeline.load_lora_weights(LORA_PATH)

# 4. Tworzenie promptu (użyj frazy, na której model się uczył)
prompt = "a photo of a dog, high resolution, detailed masterpiece, cinematic lighting"

# Zmniejszamy num_inference_steps do 20, aby przyspieszyć proces na CPU bez straty jakości
# Ustalamy stałe ziarno losowości, np. liczbę 42
generator = torch.Generator("cpu").manual_seed(42)

# Przekazujemy generator do pipeline'u
image = pipeline(prompt, num_inference_steps=20, generator=generator).images[0]


# 5. Zapisanie gotowego obrazu na dysku
image.save("wytrenowany_pies_lokalnie.png")
print("Sukces! Twój kot został zapisany jako 'wytrenowany_kot_lokalnie.png'")