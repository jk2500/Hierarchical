# README

This repository contains sample code illustrating the use of a pretrained [Stable Diffusion VAE (AutoencoderKL)](https://huggingface.co/stabilityai/sd-vae-ft-mse) as a perception module for images, combined with an RNN model for predicting next-step latent states. Additionally, an example integration with the [iGibson](https://github.com/StanfordVL/iGibson) environment is provided to show how one might collect real-time data for training the RNN on perception/action sequences in a simulated environment.

Below is an overview of each file, along with usage instructions.

---

## Repository Contents

1. **`Autoencoder.py`**  
   - Demonstrates how to load the pretrained `AutoencoderKL` model from the [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse) checkpoint.  
   - Shows how to preprocess input images to feed into the autoencoder, and then reconstruct them.  
   - Visualizes the original image vs. reconstructed image side by side.

2. **`RNN.py`**  
   - Defines an `RNNPerceptionAction` class that takes a flattened perception latent (from the VAE) and a separate action latent as input, then predicts the next perception latent state.  
   - Uses a simple LSTM architecture for sequence modeling.  
   - Includes a dummy data generator to simulate perception/action pairs for training.  
   - Contains a training loop that demonstrates how to train this RNN using synthetic data.

3. **`RNN_igibson.py`**  
   - Extends the RNN approach by integrating with the [iGibson](https://github.com/StanfordVL/iGibson) environment to collect real observation and action data.  
   - Shows how to load and encode images on-the-fly using the pretrained `AutoencoderKL`.  
   - Illustrates how to generate action latents (e.g., random or from a policy) and use them together with the perception latents to train the RNN in a real-time simulation loop.  
   - Contains a configurable training loop with iGibson environment resets, collecting sequences of observations and actions for each training batch.

4. **`my_robot_env.yaml`**  
   - An example iGibson environment configuration file specifying robot, scene, sensor, and task parameters.  
   - You can customize this file (e.g., different robots, different tasks, or different randomization settings) for your own experiments.

5. **`LICENSE`**  
   - MIT License for this repository’s code.

---

## Setup and Requirements

1. **Python environment**  
   - It is recommended to create a virtual environment (e.g., `conda` or `venv`) to manage dependencies.

2. **Install PyTorch**  
   - You need an appropriate PyTorch build for your hardware (e.g., CUDA if you have an NVIDIA GPU, MPS if you have an Apple Silicon machine).  
   - See [PyTorch’s installation instructions](https://pytorch.org/get-started/locally/) for more details.

3. **Install Hugging Face `diffusers` and `transformers`**  
   ```bash
   pip install diffusers transformers accelerate
   ```

4. **Install iGibson (for `RNN_igibson.py`)**  
   - Follow the official [iGibson installation guide](https://github.com/StanfordVL/iGibson/blob/master/docs/installation.md).  
   - Once installed, you should be able to import `igibson.envs.igibson_env`.

5. **Other common libraries**:  
   ```bash
   pip install matplotlib Pillow
   ```

---

## Instructions

### 1. Using `Autoencoder.py`

- **Goal**: Test out the pretrained VAE model and visualize how well it reconstructs an input image.  
- **Steps**:
  1. Place an image in `./Images/` (or update the `image_path` variable to your own path).  
  2. Run:
     ```bash
     python Autoencoder.py
     ```
  3. A window will pop up showing the original and reconstructed images side by side.

### 2. Using `RNN.py`

- **Goal**: Train the `RNNPerceptionAction` model with **dummy** (synthetic) data.  
- **Steps**:
  1. Make sure you have PyTorch installed and the `diffusers` library for the autoencoder.  
  2. Run:
     ```bash
     python RNN.py
     ```
  3. The script will:
     - Load the pretrained autoencoder to determine the latent dimension size.  
     - Generate random dummy data for perception latents and action latents.  
     - Train an LSTM-based model to predict the next perception latent.  
     - Print out the loss at each epoch.

### 3. Using `RNN_igibson.py`

- **Goal**: Collect real perception data from iGibson and train the RNN in an interactive environment.  
- **Steps**:
  1. Install iGibson and ensure you can import it (`import igibson`).  
  2. Edit `my_robot_env.yaml` or provide your own config file to customize the robot, scene, and tasks.  
  3. Run:
     ```bash
     python RNN_igibson.py
     ```
  4. The script will:
     - Launch an iGibson environment in **headless** mode by default.  
     - Reset the environment and capture a sequence of `rgb` frames.  
     - Preprocess and encode these frames using the autoencoder into latent representations.  
     - Generate random actions or use an existing policy.  
     - Train the RNN to predict the next perception latent.  

**Note**: The provided code is a minimal example; real-world usage will require robust data handling, and you may need to refine action spaces, environment resets, or training loops.

---

## License

This project is licensed under the [MIT License](./LICENSE). Please see the `LICENSE` file for details.

---

## Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for the pretrained VAE model.  
- [PyTorch](https://pytorch.org/) for deep learning frameworks and utilities.  
- [iGibson](https://github.com/StanfordVL/iGibson) for the simulation environment used in `RNN_igibson.py`.  
- [Stability AI](https://huggingface.co/stabilityai) for providing the Stable Diffusion VAE.  

Feel free to contribute by submitting issues or pull requests!

