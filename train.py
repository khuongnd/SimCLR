import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from model import Encoder
from utils import GaussianBlur

batch_size = 32
out_dim = 64
s = 1

color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

data_augment = transforms.Compose([transforms.ToPILImage(),
                                   transforms.RandomResizedCrop(96),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomApply([color_jitter], p=0.8),
                                   transforms.RandomGrayscale(p=0.2),
                                   GaussianBlur(),
                                   transforms.ToTensor()])

train_dataset = datasets.STL10('data', download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, drop_last=True, shuffle=True)

model = Encoder(out_dim=out_dim)
print(model)

train_gpu = False  ## torch.cuda.is_available()
print("Is gpu available:", train_gpu)
# moves the model paramemeters to gpu
if train_gpu:
    model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 3e-4)

for e in range(20):
    for step, (batch_x, _) in enumerate(train_loader):
        # print("Input batch:", batch_x.shape, torch.min(batch_x), torch.max(batch_x))
        optimizer.zero_grad()

        xis = []
        xjs = []
        for k in range(len(batch_x)):
            xis.append(data_augment(batch_x[k]))
            xjs.append(data_augment(batch_x[k]))

        # fig, axs = plt.subplots(nrows=1, ncols=6, constrained_layout=False)
        # fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=False)
        # for i_ in range(3):
        #     axs[i_, 0].imshow(xis[i_].permute(1, 2, 0))
        #     axs[i_, 1].imshow(xjs[i_].permute(1, 2, 0))
        # plt.show()

        xis = torch.stack(xis)
        xjs = torch.stack(xjs)
        # print("Transformed input stats:", torch.min(xis), torch.max(xjs))

        _, zis = model(xis)  # [N,C]
        # print(his.shape, zis.shape)

        _, zjs = model(xjs)  # [N,C]
        # print(hjs.shape, zjs.shape)

        # positive pairs
        l_pos = torch.bmm(zis.view(batch_size, 1, out_dim), zjs.view(batch_size, out_dim, 1)).view(batch_size, 1)
        assert l_pos.shape == (batch_size, 1)  # [N,1]
        l_neg = []

        for i in range(zis.shape[0]):
            mask = np.ones(zjs.shape[0], dtype=bool)
            mask[i] = False
            negs = torch.cat([zjs[mask], zis[mask]], dim=0)  # [2*(N-1), C]
            l_neg.append(torch.mm(zis[i].view(1, zis.shape[-1]), negs.permute(1, 0)))

        l_neg = torch.cat(l_neg)  # [N, 2*(N-1)]
        assert l_neg.shape == (batch_size, 2 * (batch_size - 1)), "Shape of negatives not expected." + str(l_neg.shape)
        # print("l_neg.shape -->", l_neg.shape)

        logits = torch.cat([l_pos, l_neg], dim=1)  # [N,K+1]
        # print("logits.shape -->",logits.shape)

        labels = torch.zeros(batch_size, dtype=torch.long)

        if train_gpu:
            labels = labels.cuda()

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        print("Step {}, Loss {}".format(step, loss))

torch.save(model.state_dict(), './model/checkpoint.pth')