import os
import cv2
import numpy as np
import csv
import pickle
import zstandard as zstd
import torch
from cat_autoencoder import CatAutoencoder

#calculates and writes stats to csv file
def writeStatsToCSV(csv_filename, image_name, ratio, original_image_array, compressed_bin_filename):
    file_exists = os.path.isfile(csv_filename)
    
    # 1. calculates file sizes
    raw_size_bytes = original_image_array.nbytes
    compressed_size_bytes = os.path.getsize(compressed_bin_filename)
    
    # Calculate sizes in KB and the compression ratio
    raw_kb = raw_size_bytes / 1024
    compressed_kb = compressed_size_bytes / 1024
    compression_ratio = raw_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0
    
    # data structure
    stats_data = {
        "Image_Name": image_name,
        "Ratio": ratio,
        "Original_KB": round(raw_kb, 3),
        "Compressed_KB": round(compressed_kb, 3),
        "Compression_Ratio": round(compression_ratio, 3)
    }
    
    # 3. write portion
    with open(csv_filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_data.keys())
        
        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(stats_data)
    
    print(f"Stats successfully appended to {csv_filename}")

# autoencoder compression
def compress_with_nn(image_path, model, output_bin_path="nn_compressed.bin"):
    # 1. Load and prepare the image (Resize to 128x128 for this specific network)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Convert to PyTorch Tensor (Normalize 0 to 1, and reshape to C x H x W)
    tensor_img = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    tensor_img = tensor_img.unsqueeze(0) # Add a batch dimension: [1, 3, 128, 128]
    
    # 3. Compress! (Pass ONLY through the encoder)
    model.eval()
    with torch.no_grad():
        latent_space = model.encoder(tensor_img)
    
    # 4. Save to binary file
    # We convert to float16 to literally halve the file size on disk!
    latent_numpy = latent_space.squeeze(0).numpy().astype(np.float16) 
    latent_numpy.tofile(output_bin_path)
    
    # Return the file size to write to your CSV
    return os.path.getsize(output_bin_path) / 1024.0  # Return size in KB

# autoencoder decompression
def decompress_with_nn(bin_path, model):
    # 1. Load the binary file back into a NumPy array
    latent_numpy = np.fromfile(bin_path, dtype=np.float16)
    
    # 2. Reshape back to the latent dimensions (8 channels, 16x16)
    latent_numpy = latent_numpy.reshape(8, 16, 16) #8,16,16 dla v1, 8,8,8 dla v2
    
    # 3. Convert back to PyTorch Tensor
    latent_tensor = torch.tensor(latent_numpy, dtype=torch.float32).unsqueeze(0)
    
    # 4. Decompress! (Pass ONLY through the decoder)
    model.eval()
    with torch.no_grad():
        reconstructed_tensor = model.decoder(latent_tensor)
    
    # 5. Convert back to an OpenCV Image (BGR, 0-255)
    reconstructed_img = reconstructed_tensor.squeeze(0).permute(1, 2, 0).numpy()
    reconstructed_img = (reconstructed_img * 255).astype(np.uint8)
    reconstructed_bgr = cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR)
    
    return reconstructed_bgr

image_filename="images/grey_cat.png"
decompressed_name = "images/grey_cat_nn.png"
compressed_filename = "compressed_grey.bin"
# with open(compressed_filename, "wb") as f:
#     clean_package = {
#         "Y": compressed.Y,
#         "Cr": compressed.Cr,
#         "Cb": compressed.Cb,
#         "QY": compressed.QY,
#         "QC": compressed.QC,
#         "chroma_ratio": compressed.chroma_ratio
#     }
#     pickle.dump(clean_package, f)
model = CatAutoencoder()
model.load_state_dict(torch.load("cat_autoencoder_v1.pth", map_location=torch.device('cpu')))
model.eval()
original_image = cv2.imread(image_filename)
resized_image = cv2.resize(original_image, (512, 512), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
compress_with_nn(image_path=image_filename, model=model, output_bin_path=compressed_filename)
writeStatsToCSV("compression_results_nn.csv", image_filename, "Autoencoder", resized_image, compressed_filename)

decompressed_image = decompress_with_nn(compressed_filename, model)
width, height = original_image.shape[:2]
decompressed_image =  cv2.resize(decompressed_image, (height, width) , dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
cv2.imwrite(decompressed_name, decompressed_image)