import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

device = torch.device("mps")

def flatten(x):
    """Flatten the tensor."""
    return x.view(x.size(0), -1)

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST("./data/", train=True, download=True, transform=transform),
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        MNIST("./data/", train=False, download=True, transform=transform),
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


# def overlay_y_on_x(x, y):
#     x_ = x.clone()
#     x_[:, :, :10, :] *= 0.0
#     x_[range(x.shape[0]), :, y, :] = x.max()
#     return x_

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_ = torch.flatten(x_, start_dim=1)
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    x_ = torch.reshape(x_[:, None, :], x.shape)
    return x_

class Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        self.layers += [Layer_Conv2d(1, 6, 5, 1, 2).to(device)]
        self.layers += [Layer_Pool_Conv2d(6, 16, 5, 1, 0).to(device)]
        self.layers += [Layer_Pool_Conv2d(16, 120, 5, 1, 0).to(device)]
        self.layers += [Layer(120, 84).to(device)]
        # self.layers += [Layer(84, 10).to(device)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                h2 = h.pow(2).mean(1)
                h2 = flatten(h).view(x.shape[0], -1).mean(1)
                goodness += [h2]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print("training layer", i, "...")
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(params=self.fc.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 60


    def forward(self, x):
        x = flatten(x)
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        res = self.fc(x_direction)
        return self.relu(res)

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(
                1
                + torch.exp(
                    torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
                )
            ).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class Layer_Conv2d(Layer):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        stride,
        padding,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.opt = Adam(params=self.conv.parameters(), lr=0.03)
        

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        res = self.conv(x_direction)
        return self.relu(res)


class Layer_Pool_Conv2d(Layer):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        stride,
        padding,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        params = list(self.pool.parameters()) + list(self.conv.parameters())
        self.opt = Adam(params=params, lr=0.03)

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        res = self.pool(x_direction)
        res = self.conv(res)
        return self.relu(res)


if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = Net([1, 6, 16, 120, 84])
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    net.train(x_pos, x_neg)

    print("train error:", 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)

    print("test error:", 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
