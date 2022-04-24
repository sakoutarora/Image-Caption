import torch.nn as nn
import torch
from datasetLoad import loader
import torchvision.transforms as transform
from model import CNNtoRNN
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm


def train():
    tr = transform.Compose([
        transform.Resize((356, 356)),
        transform.RandomCrop(299, 229),
        transform.ToTensor()
    ])
    load, dataset = loader(transform=tr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    # Hyperparameter
    embed_size = 256
    hidden_size = 256
    vocab_len = dataset.vocab_len
    num_layers = 1
    learning_rate = 3e-3
    epochs = 10

    model = CNNtoRNN(embed_size, vocab_len, hidden_size, num_layers)
    model = model.to(device)
    crit = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        for imgs, caption in tqdm(load):
            imgs = imgs.to(device)
            caption = caption.to(device)
            output = model(imgs, caption[:-1])
            loss = crit(output.reshape(-1, output.shape[2]), caption.reshape(-1).long())
            writer.add_scalar("Loss/train", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    writer.flush()


if __name__ == "__main__":
    train()

