import torch
torch.cuda.current_device()
import torch.nn as nn
from torchvision import datasets ,transforms
import torchvision
from matplotlib import pyplot as plt
import numpy as np
# from google.colab.patches import cv2_imshow
import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
# from tensorflow import summary
from torchvision.utils import make_grid
import os, cv2, argparse
import torch.backends.cudnn as cudnn


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    EPOCH = 50                # 全部data訓練10次
    BATCH_SIZE = 128           # 每次訓練隨機丟50張圖像進去
    LR = args.lr              # learning rate
    # input_shape = (1,3,48,48)

    #Load data set
    print('==> Preparing data..')
    train_data = datasets.ImageFolder(
        'Trail_dataset2/train_data',
        transform = transforms.Compose([transforms.Resize(size=(32, 32)),  # range [0, 255] -> [0.0,1.0] -> [-1.0,1.0]
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.2, 0.2, 0.2)),
                                        ])                         
    )

    test_data = datasets.ImageFolder(
        'Trail_dataset2/test_data',
        transform = transforms.Compose([transforms.Resize(size=(32, 32)),  # range [0, 255] -> [0.0,1.0] -> [-1.0,1.0]
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.2, 0.2, 0.2)),
                                        ])                         
    )

    # Check data size


    # Pytorch DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=False)

    #show labels
    print("Classes:", train_data.classes)
    print("Classes and index", train_data.class_to_idx)

    #check availability of gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    """""

    This Part You Have To Build Your Own Model

    For example:
    class CNN_Model(nn.Module)


    And at the end, you have to save your trained weights to pth format
    you can use function torch.save()

    """""

    ## Build model 
    # Create CNN Model
    from dla import DLA
    from dla_simple import SimpleDLA

    def test():
        net = DLA()
        # net = SimpleDLA()
        print(net)
        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        print(y.size())

    print('==> Building model..')
    model = DLA().to(device)
    if device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

    if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            model.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

    test()
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


    # Training
    from utils import progress_bar
    def train(epoch):
        print('\nEpoch: %d / %d     lr = %.5f' % (epoch+1, EPOCH, scheduler.get_lr()[0]))
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        loss_his = train_loss/(batch_idx+1)
        acc_his = 100.*correct/total
        return loss_his, acc_his


    def test(epoch):
        global best_acc
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        loss_his = test_loss/(batch_idx+1)
        acc_his = 100.*correct/total

        # Save best checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('run/epochs/checkpoint'):
                os.mkdir('run/epochs/checkpoint')
            torch.save(state, './run/epochs/checkpoint/ckpt.pth')
            best_acc = acc

        return loss_his, acc_his


    def fit_model(num_epochs):
        # Traning the Model
        #history-like list for store loss & acc value
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []

        for epoch in range(start_epoch, start_epoch+num_epochs):
            train_loss, train_acc = train(epoch)
            test_loss, test_acc = test(epoch)
            scheduler.step()
            training_loss.append(train_loss)
            training_accuracy.append(train_acc)
            validation_loss.append(test_loss)
            validation_accuracy.append(test_acc)

        return training_loss, training_accuracy, validation_loss, validation_accuracy

    # Training
    training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(num_epochs=EPOCH)

    # # Save directory
    if not os.path.exists("run/epochs"):
        os.makedirs("run/epochs")

    # visualization
    plt.figure()
    plt.plot(range(EPOCH), training_loss, 'b-', label='Train')
    plt.plot(range(EPOCH), validation_loss, 'r-', label='Val')
    plt.title('Training & Validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('run/epochs/History_loss.png')

    plt.figure()
    plt.plot(range(EPOCH), training_accuracy, 'b-', label='Train')
    plt.plot(range(EPOCH), validation_accuracy, 'r-', label='Val')
    plt.title('Training & Validation accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('run/epochs/History_acc.png')

    # plt.tight_layout()
    plt.show()