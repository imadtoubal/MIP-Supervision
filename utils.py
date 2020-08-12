import torch
from torch import nn
import torch.optim as optim


def train(net, criterion=nn.CrossEntropyLoss(), lr=0.0001, epochs=100, device='cuda'):
    net = net.to(device)
    criterion = criterion.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_dice = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.long().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, yi, yj, yk = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # dice coeff
            dice = dice_coeff(outputs[:, 1, :, :, :], labels)

            # print statistics
            running_loss += loss.item()
            running_dice += dice.item()

            print(
                f'[{epoch+1}, {i+1}] loss: {loss.item():.3} \t dice: {dice.item():.3}')

            writer.add_scalar('Loss/Training loss',
                              running_loss/len(trainset), epoch)
            writer.add_scalar('DICE/Training dice',
                              running_dice/len(trainset), epoch)

        with torch.no_grad():
            val_loss = 0.0
            val_dice = 0.0
            for i, data in enumerate(valloader, 0):
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.long().to(device)

                outputs, yi, yj, yk = net(inputs)

                loss = criterion(outputs, labels)
                dice = dice_coeff(outputs[:, 1, :, :, :], labels)

                val_loss += loss.item()
                val_dice += dice.item()

                if i == 0:
                    outs = outputs[:, 1, :, :, :].transpose(
                        0, 3).transpose(1, 3)[::4]
                    grid_outs = torchvision.utils.make_grid(outs)
                    writer.add_image('Segmentation result', grid_outs, epoch)

        print(
            f'[{epoch+1}] val loss: {val_loss/len(valset):.4} \t val dice: {val_dice/len(valset):.3}')

        writer.add_scalar('Loss/Validation loss', val_loss/len(valset), epoch)
        writer.add_scalar('DICE/Validation dice', val_dice/len(valset), epoch)

    print('Finished Training')
