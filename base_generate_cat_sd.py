import torch
from diffusers import StableDiffusionPipeline

# 1. Ładowanie czystego modelu bazowego Stable Diffusion 1.5
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
print("Ładowanie CZYSTEGO modelu bazowego... (używa plików z pamięci podręcznej)")

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Wymuszamy uruchomienie na procesorze (CPU)
pipeline = pipeline.to("cpu")

# Optymalizacja pamięci RAM dla stabilnego działania
pipeline.enable_attention_slicing()

# --- POMIJAMY ŁADOWANIE wag LoRA (dzięki temu mamy wersję fabryczną) ---

# 2. Identyczny prompt dla sprawiedliwego porównania
prompt = "a photo of a cat, high resolution, detailed masterpiece, cinematic lighting"
# Ustalamy stałe ziarno losowości, np. liczbę 42
generator = torch.Generator("cpu").manual_seed(42)

# Przekazujemy generator do pipeline'u
image = pipeline(prompt, num_inference_steps=20, generator=generator).images[0]

# 3. Zapisanie czystego kota pod inną nazwą
image.save("kot_bazowy_lokalnie.png")
print("Sukces! Czysty kot został zapisany jako 'czysty_kot_bazowy.png'")