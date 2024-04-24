import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparameters
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4

# Dataset
dataset = datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=BATCH_SIZE)
model = VariationalAutoEncoder(input_dim=INPUT_DIM,h_dim=H_DIM,z_dim=Z_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
loss_fn = nn.MSELoss(reduction='sum') # the output will be summed

# trainig loop
def train():

    results = {"total_loss": [],
        "reconstruction_loss": [],
        "kl_div": []
    }

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_dataloader))
        print(f"Epoch {epoch + 1}\n")
        for i, (x, _) in loop:
            # forward pass
            x = x.to(device).view(x.shape[0],INPUT_DIM) # view() reshapes the tensor without copying memory, similar to numpy's reshape()
            x_reconstructed, mu, sigma = model(x)

            # loss
            reconstruction_loss = loss_fn(x_reconstructed,x) # pushes to reconstruct the image
            kl_div = -0.5*torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) # pushes towards standard Guassian

            # backprop
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item()) # for additional stats
            #print(loss.cpu())
        results["total_loss"].append(loss.cpu().detach().numpy())
        results["reconstruction_loss"].append(reconstruction_loss.cpu().detach().numpy())
        results["kl_div"].append(kl_div.cpu().detach().numpy())

    return results

def plot_loss(results):
    loss = results['total_loss']
    reconstruction_loss = results["reconstruction_loss"]
    kl_div = results['kl_div']

    # Figure out how many epochs there were
    epochs = range(len(results['total_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot total_loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, label='total_loss')
    plt.title('total_loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot reconstruction_loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, reconstruction_loss, label='reconstruction_loss')
    plt.title('reconstruction_loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot kl_div
    plt.subplot(1, 3, 3)
    plt.plot(epochs, kl_div, label='kl_div')
    plt.title('kl_div')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()

if __name__=='__main__':
    results = train()
    model_save_path = "models"
    torch.save(model.state_dict(), "models/vae_mse.pth")
    print('model saved')
    plot_loss(results)
