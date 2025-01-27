import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Resize, Compose
from diffusers import AutoencoderKL
from PIL import Image

# iGibson imports (make sure iGibson is installed)
# pip install igibson
from igibson.envs.igibson_env import iGibsonEnv
import numpy as np
import os

# -------------------------------
# 1. Define the RNN model
# -------------------------------
class RNNPerceptionAction(nn.Module):
    def __init__(self, latent_dim_perception, latent_dim_action, hidden_dim, num_layers=1):
        """
        RNN to predict the next perception latent space.
        Args:
        - latent_dim_perception: Dimensionality of flattened perception latent representation.
        - latent_dim_action: Dimensionality of action latent representation.
        - hidden_dim: Number of hidden units in the RNN.
        - num_layers: Number of RNN layers.
        """
        super(RNNPerceptionAction, self).__init__()

        # Combine perception and action latents
        self.input_dim = latent_dim_perception + latent_dim_action

        # RNN layer
        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected layer to map RNN output to the latent space
        self.fc = nn.Linear(hidden_dim, latent_dim_perception)

    def forward(self, perception_latent, action_latent, hidden_state=None):
        """
        Forward pass through the RNN.
        Args:
        - perception_latent: Tensor of shape [batch_size, seq_length, latent_dim_perception].
        - action_latent: Tensor of shape [batch_size, seq_length, latent_dim_action].
        - hidden_state: Initial hidden state for the RNN (optional).
        Returns:
        - output: Tensor of shape [batch_size, seq_length, latent_dim_perception].
        - hidden_state: Final hidden state of the RNN.
        """
        # Concatenate perception and action latents
        rnn_input = torch.cat((perception_latent, action_latent), dim=-1)  # [batch_size, seq_length, input_dim]

        # Pass through RNN
        rnn_output, hidden_state = self.rnn(rnn_input, hidden_state)  # [batch_size, seq_length, hidden_dim]

        # Map RNN output to latent space prediction
        output = self.fc(rnn_output)  # [batch_size, seq_length, latent_dim_perception]

        return output, hidden_state


# -------------------------------
# 2. Load the pretrained Autoencoder
# -------------------------------
autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
autoencoder.to(device)

# Image preprocessing (iGibson outputs are typically [H, W, 3] in 0..255)
image_size = 256
preprocess = Compose([
    Resize((image_size, image_size)),
    ToTensor(),
    lambda x: (x - 0.5) * 2  # Normalize to [-1, 1]
])

# Helper: encode a batch of images into latents
def encode_images(images, autoencoder):
    """
    images: [batch_size, 3, H, W] in range [-1, 1].
    Returns: Flattened latents of shape [batch_size, latent_dim].
    """
    with torch.no_grad():
        latents = autoencoder.encode(images).latent_dist.sample()  # => [B, 4, H/8, W/8]
        latents = latents.view(latents.size(0), -1)  # flatten => [B, 4*(H/8)*(W/8)]
    return latents

# -------------------------------
# 3. iGibson environment setup
# -------------------------------
# Provide a valid config file from iGibson. The example below may need changing.
config_file = "./my_robot_env.yaml"  # or another valid config
env = iGibsonEnv(config_file=config_file, mode="headless")

# -------------------------------
# 4. Hyperparameters
# -------------------------------
# The Autoencoder output has shape [B, autoencoder.config.latent_channels, 32, 32] for a 256x256 input,
# which is latent_channels * 32 * 32 elements when flattened.
latent_dim_perception = autoencoder.config.latent_channels * (image_size // 8) ** 2
latent_dim_action = 8   # Example action dimension
hidden_dim = 64
num_layers = 1
learning_rate = 0.001
num_epochs = 20
seq_length = 10
batch_size = 4          # We'll collect "batch_size" sequences each epoch (optional)

# -------------------------------
# 5. Initialize the RNN model
# -------------------------------
model = RNNPerceptionAction(latent_dim_perception, latent_dim_action, hidden_dim, num_layers).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 6. iGibson data collection
# -------------------------------
def collect_igibson_sequence(env, seq_length, preprocess, autoencoder, device):
    """
    Collects a single sequence of (perception_latent, action) from an iGibson environment.
    Returns:
      perception_latents: [1, seq_length, latent_dim_perception]
      action_latents:     [1, seq_length, latent_dim_action]
    """
    obs = env.reset()
    latents_list = []
    actions_list = []

    for step in range(seq_length):
        # 1) Extract the RGB image from obs
        rgb = obs["rgb"]  # shape [H, W, 3], typically uint8

        # 2) Preprocess => [1, 3, 256, 256], then encode => [1, latent_dim_perception]
        pil_img = Image.fromarray(rgb)
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        latents = encode_images(img_tensor, autoencoder)

        latents_list.append(latents)

        # 3) Random action (for demonstration). Adjust to match your actual action space.
        #    For example, if the action space is 2D or 4D, etc.
        action_dim = latent_dim_action
        action = np.random.uniform(-1.0, 1.0, size=(action_dim,))
        actions_list.append(action)

        # 4) Step environment
        obs, reward, done, info = env.step(action)

        # If the episode ends early, you may want to break or reset
        if done:
            break

    # Convert to Tensors: shape [seq_length, latent_dim]
    perception_latents = torch.cat(latents_list, dim=0)              # [seq_length, latent_dim_perception]
    action_latents = torch.tensor(actions_list, dtype=torch.float, device=device)  # [seq_length, latent_dim_action]

    # Reshape to [1, seq_length, latent_dim]
    perception_latents = perception_latents.unsqueeze(0)
    action_latents = action_latents.unsqueeze(0)

    return perception_latents, action_latents

# -------------------------------
# 7. Training loop
# -------------------------------
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Collect sequences from iGibson. Here we collect `batch_size` sequences
    # and combine them so we can simulate a mini-batch.
    all_perception = []
    all_actions = []

    for _ in range(batch_size):
        perc_lat, act_lat = collect_igibson_sequence(env, seq_length, preprocess, autoencoder, device)
        all_perception.append(perc_lat)
        all_actions.append(act_lat)

    # Combine all into a single batch
    # all_perception: list of [1, seq_length, latent_dim_perception]
    perception_latent = torch.cat(all_perception, dim=0)  # [batch_size, seq_length, latent_dim_perception]
    action_latent = torch.cat(all_actions, dim=0)         # [batch_size, seq_length, latent_dim_action]

    # Next perception latents (shift by 1 along the seq_length dimension)
    next_perception_latent = torch.roll(perception_latent, shifts=-1, dims=1)

    # Forward pass
    predicted_latent, _ = model(perception_latent, action_latent)

    # Compute loss
    loss = criterion(predicted_latent, next_perception_latent)

    # Backward + Optimize
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")
