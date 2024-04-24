import torch
import torchvision.datasets as datasets
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
import os


# configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparameters
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20

dataset = datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
loaded_model = VariationalAutoEncoder(input_dim=INPUT_DIM,h_dim=H_DIM,z_dim=Z_DIM).to(device)
loaded_model.load_state_dict(torch.load("models/vae_mse.pth"))

def inference(digit, num_examples):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = loaded_model.encoder(images[d].to(device).view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = loaded_model.decoder(z)
        out = out.view(-1, 1, 28, 28)
        if not os.path.exists('test_mse'):
            os.makedirs('test_mse')
        save_image(out, f"test_mse/generated_{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=5)