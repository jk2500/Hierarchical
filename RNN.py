import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Resize, Compose
from diffusers import AutoencoderKL
from PIL import Image

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

# Load the pretrained AutoencoderKL
autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
autoencoder.to(device)

# Preprocessing pipeline for images
image_size = 256
preprocess = Compose([
    Resize((image_size, image_size)),
    ToTensor(),
    lambda x: (x - 0.5) * 2  # Normalize to [-1, 1]
])

# Hyperparameters
latent_dim_perception = autoencoder.config.latent_channels * (image_size // 8) ** 2  # Flattened latent dimension
latent_dim_action = 8      # Example action latent dimension
hidden_dim = 64            # Hidden state size of RNN
num_layers = 1             # Number of RNN layers
learning_rate = 0.001
num_epochs = 20
batch_size = 32
seq_length = 10

# Initialize the RNN model
model = RNNPerceptionAction(latent_dim_perception, latent_dim_action, hidden_dim, num_layers)
model.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy dataset generator
def generate_dummy_data(batch_size, seq_length, latent_dim_perception, latent_dim_action):
    """Generates dummy sequential perception latents, action latents, and next perception latents."""
    perception_latent = torch.randn(batch_size, seq_length, latent_dim_perception)
    action_latent = torch.randn(batch_size, seq_length, latent_dim_action)
    next_perception_latent = torch.roll(perception_latent, shifts=-1, dims=1)  # Shift perception latents
    return perception_latent, action_latent, next_perception_latent

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Generate dummy data (replace with real data loading in production)
    perception_latent, action_latent, next_perception_latent = generate_dummy_data(
        batch_size, seq_length, latent_dim_perception, latent_dim_action
    )

    perception_latent = perception_latent.to(device)
    action_latent = action_latent.to(device)
    next_perception_latent = next_perception_latent.to(device)

    # Forward pass
    predicted_latent, _ = model(perception_latent, action_latent)

    # Compute loss
    loss = criterion(predicted_latent, next_perception_latent)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")
