import os
import cv2
import numpy as np
import scipy.fftpack
import zlib
import csv
import pickle
import zstandard as zstd
import torch
from cat_autoencoder import CatAutoencoder

#Class used for keeping the compressed image information
class container:
    Y=np.array([])
    Cb=np.array([])
    Cr=np.array([])
    chroma_ratio="4:4:4"
    QY=np.ones((8,8))
    QC=np.ones((8,8))
    shape=(0,0,3)

#DCT - dyskretna transformata cosinusowa, gotowiec
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')
#Note: to make it easier, we assume the size of all images is a multiplication of 8 or 16


#Changing the way that color information is kept from RGB into YCbCr.

#Y - luminance 
#Cb - blue-difference chroma
#Cr - red-difference chroma

def compressColors(original_image, ratio = "4:4:4",QY=np.ones((8,8)),QC=np.ones((8,8))):
    # RGB -> YCrCb
    #adding info given to the class
    compressed= container()
    compressed.chroma_ratio=ratio
    compressed.QY = QY
    compressed.QC = QC
    new_image = cv2.cvtColor(original_image,cv2.COLOR_RGB2YCrCb).astype(int)
    compressed.Y, compressed.Cr, compressed.Cb = cv2.split(new_image)

    return compressed

def decompressColors(compressed_data):
    reconstructed_image = cv2.merge([compressed_data.Y, compressed_data.Cr, compressed_data.Cb])
    final_image = cv2.cvtColor(reconstructed_image.astype(np.uint8),cv2.COLOR_YCrCb2RGB)

    return final_image


#Take a layer and turn it into a vector, using the right transformations for compression

#Ziggzagging to get a vector from the given block or a block from a vector
def zigzag(A):
    #the numbers follow the line of ziggzagging
    template= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    #reverse ziggzagging
    if len(A.shape)==1:
        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]
    #Ziggzagging
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]
    return B


def compressBlock(block, Q):
    #first, discrete cosinine transform
    d = dct2(block)
    #then compression
    qd=np.round(d/Q).astype(int)
    #finally, zigzagging
    v = zigzag(qd)
    return v

def decompressBlock(block, Q):
    qd = zigzag(block)
    pd=qd*Q
    matrix = idct2(pd)
    return matrix

#L -> compressed layer, S-> vector of values
def CompressLayer(L,Q):
    S=np.array([])
    for w in range(0,L.shape[0],8):
        for k in range(0,L.shape[1],8):
            block=L[w:(w+8),k:(k+8)]
            S=np.append(S, compressBlock(block,Q))
    return S

def DecompressLayer(S,Q, ratio):
    num_blocks = len(S) // 64
    total_pixels = num_blocks * 64
    if total_pixels == 262144 or ratio == "4:2:0" or ratio == "4:4:4" or ratio == "4:1:0:0:0" or ratio == "final":
        a = int(np.sqrt(total_pixels))
        height, width = a, a
    elif ratio == "4:2:2":
        a = int(np.sqrt(total_pixels / 2))
        height, width = 2 * a, a
    elif ratio == "4:4:0":
        a = int(np.sqrt(total_pixels / 2))
        height, width = a, 2 * a
    elif ratio == "4:1:0":
        a = int(np.sqrt(total_pixels / 2))
        height, width = 2 * a, a
    elif ratio == "8:1:0":
        a = int(np.sqrt(total_pixels/4))
        height, width = 4*a, a
    elif ratio == "8:1:0:0:0":
        a = int(np.sqrt(total_pixels/2))
        height, width = 2*a, a
    else:
        a = int(np.sqrt(total_pixels / 2))
        height, width = a, 2 * a
    L = np.zeros((height, width), dtype=np.float32)
    blocks_per_row = width // 8
    blocks_per_col = height // 8
    for idx in range(num_blocks):
        start = idx * 64
        vector = S[start:start + 64]
        row = idx // blocks_per_row
        col = idx % blocks_per_row
        w = row * 8
        k = col * 8
        L[w:(w+8),k:(k+8)]=decompressBlock(vector,Q)
    return L

#Chroma subsampling

#L-> layer
def chromaSubsampling(L, ratio="4:4:4"):
    if ratio == "4:4:4":
        B=L #No compression
    elif ratio == "4:2:2":
        B=L[::,::2] #every other column
    elif ratio == "4:4:0":
        B=L[::2,::] #every other line
    elif ratio== "4:2:0":
        B=L[::2,::2] #every other column and every other line
    elif ratio == "4:1:0":
        B=L[::2,::4] #every other line and every four columns
    elif ratio == "8:1:0":
        B=L[::2,::8] #every other line and every eight columns
    elif ratio == "4:1:0:0:0":
        B=L[::4,::4] #every four lines and every four columns
    elif ratio == "8:1:0:0:0":
        B=L[::4,::8] #every four lines and every eight columns
    elif ratio == "final":
        B=L[::8,::8] #every eight lines and every eight columns
    else:
        print("Wrong ratio, doing 4:4:4")
        B=L
    return B

def chromaResampling(L, ratio="4:4:4"):
    if ratio=="4:4:4":
        B=L #there has been no compression
    elif ratio=="4:2:2":
        B = np.repeat(L, repeats=2, axis=1) #duplicating every column
    elif ratio=="4:4:0":
        B = np.repeat(L, repeats=2, axis=0)
        print(B.shape)
    elif ratio == "4:2:0":
        B = np.repeat(L,repeats=2, axis=0)
        B = np.repeat(B, repeats=2, axis=1)
    elif ratio == "4:1:0":
        B = np.repeat(L, repeats=4, axis=1) 
        B = np.repeat(B, repeats=2, axis=0)
        print(B.shape)
    elif ratio == "8:1:0":
        B = np.repeat(L, repeats=8, axis=1) 
        B = np.repeat(B, repeats=2, axis=0)
        print(B.shape)
    elif ratio == "4:1:0:0:0":
        print("HERE")
        B = np.repeat(L, repeats=4, axis=1) 
        B = np.repeat(B, repeats=4, axis=0)
        print(B.shape)
    elif ratio == "8:1:0:0:0":
        print("HERE")
        B = np.repeat(L, repeats=8, axis=1) 
        B = np.repeat(B, repeats=4, axis=0)
        print(B.shape)
    elif ratio == "final":
        print("HERE")
        B = np.repeat(L, repeats=8, axis=1) 
        B = np.repeat(B, repeats=8, axis=0)
        print(B.shape)
    else:
        print("Wrong ratio, doing 4:4:4")
        B=L
    return B

#huffman + zl77 from zlib
def applyEntropyCoding(layer):
    # Convert numpy array to 16-bit integers (to save space), then to raw bytes
    layer_bytes = layer.astype(np.int16).tobytes()
    # Compress the bytes using zlib (Deflate/Huffman)
    #return zlib.compress(layer_bytes, level=9)
    compressor = zstd.ZstdCompressor(level=22) 
    return compressor.compress(layer_bytes)


def removeEntropyCoding(compressed_bytes):
    # Decompress the bytes back to their original state
    #decompressed_bytes = zlib.decompress(compressed_bytes)
    
    decompressor = zstd.ZstdDecompressor()
    decompressed_bytes = decompressor.decompress(compressed_bytes)
    return np.frombuffer(decompressed_bytes, dtype=np.int16).astype(float)



#Final compression and decompression

def compressAll(original_image, ratio = "4:4:4",QY=np.ones((8,8)),QC=np.ones((8,8))):
    #First, change color scheme
    img = compressColors(original_image, ratio, QY, QC)
    #chroma subsampling for chroma layers only
    img.Cb = chromaSubsampling(img.Cb, ratio)
    img.Cr = chromaSubsampling(img.Cr, ratio)
    #Compressing each layer
    img.Y=CompressLayer(img.Y,img.QY)
    img.Cr=CompressLayer(img.Cr,img.QC)
    img.Cb=CompressLayer(img.Cb,img.QC)
    #tu może do dodania kompresja strumieniowa (algorytm Hoffmana)
    img.Y = applyEntropyCoding(img.Y)
    img.Cr = applyEntropyCoding(img.Cr)
    img.Cb = applyEntropyCoding(img.Cb)
    return img

def decompressAll(compressed_image):
    compressed_image.Y = removeEntropyCoding(compressed_image.Y)
    compressed_image.Cr = removeEntropyCoding(compressed_image.Cr)
    compressed_image.Cb = removeEntropyCoding(compressed_image.Cb)

    compressed_image.Y = DecompressLayer(compressed_image.Y, compressed_image.QY, ratio = "4:4:4")
    compressed_image.Cr = DecompressLayer(compressed_image.Cr, compressed_image.QC, compressed_image.chroma_ratio)
    compressed_image.Cb = DecompressLayer(compressed_image.Cb, compressed_image.QC, compressed_image.chroma_ratio)

    compressed_image.Cb = chromaResampling(compressed_image.Cb, compressed_image.chroma_ratio)
    compressed_image.Cr = chromaResampling(compressed_image.Cr, compressed_image.chroma_ratio)
    img = decompressColors(compressed_image)
    return img


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
    latent_numpy = latent_numpy.reshape(8, 16, 16)
    
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

#Tests
#4:4:4 4:4:0 4:2:2 4:2:0 4:1:0 
# 4:1:0:0:0 8:1:0 8:1:0:0:0 final
image_filename="images/red_cat.png"
ratio="final"
decompressed_name = "images/red_cat_nn.png"
model = CatAutoencoder()
model.load_state_dict(torch.load("cat_autoencoder_v1.pth", map_location=torch.device('cpu')))
model.eval()
original_image = cv2.imread(image_filename)
#image needs to be square shaped and divisible by 8
resized_image = cv2.resize(original_image, (512, 512), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)

# cv2.imshow("Original image", original_image)

#values from JPEG standard
QY= np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 36, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
        ])
QC= np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        ])

#neutral, only changing from float to int
QN= np.ones((8,8))


#compressed = compressAll(resized_image, ratio=ratio, QY=QY, QC=QC)

compressed_filename = "compressed_cat.bin"
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
compress_with_nn(image_path=image_filename, model=model, output_bin_path=compressed_filename)

# 3. WRITE TO CSV (The function does all the calculating!)
writeStatsToCSV("compression_results_nn.csv", image_filename, "Autoencoder", resized_image, compressed_filename)

decompressed_image = decompress_with_nn(compressed_filename, model)

#decompressed_image = decompressAll(compressed)
#width, height = original_image.shape[:2]
#decompressed_image = cv2.resize(decompressed_image, (height, width) , dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)

#cv2.imshow("After compression and decompression", decompressed_image)
 
# Wait for a key press before closing the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(decompressed_name, decompressed_image)