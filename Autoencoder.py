import torch
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose
from diffusers import AutoencoderKL
from PIL import Image
import matplotlib.pyplot as plt

# Load the pretrained AutoencoderKL model
model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# Ensure the model is on the correct device (e.g., GPU or MPS for Apple Silicon)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Preprocessing pipeline
image_size = 256  # Resize image to 256x256
preprocess = Compose([
    Resize((image_size, image_size)),
    ToTensor(),
    lambda x: (x - 0.5) * 2,  # Normalize to [-1, 1] as required by AutoencoderKL
])

# Postprocessing pipeline
postprocess = Compose([
    lambda x: (x / 2 + 0.5).clamp(0, 1),  # De-normalize from [-1, 1] to [0, 1]
    ToPILImage(),
])

# Load an input image
image_path = "./Images/0.png"  # Replace with your image path
input_image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(input_image).unsqueeze(0).to(device)  # Add batch dimension

# Encode and decode the image using the AutoencoderKL
with torch.no_grad():
    latents = model.encode(input_tensor).latent_dist.sample()  # Sample latent space
    print(latents.shape)
    reconstructed_tensor = model.decode(latents).sample  # Decode to reconstruct the image

# Convert the reconstructed tensor to an image
reconstructed_image = postprocess(reconstructed_tensor.squeeze(0).cpu())

# Display the input and reconstructed images side by side
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(input_image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image)
plt.axis("off")

plt.tight_layout()
plt.show()
