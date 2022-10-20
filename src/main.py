import argparse, random, copy
import os, time
from datetime import datetime
import torch
import pickle

from dataset import * 
from model import * 

def dataloader(full_dataset, args): 
    train_size = int(args.train_val_split * args.num_samples)
    val_size = args.num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    trainloader = DataLoader(train_dataset,
                            num_workers = args.num_workers, 
                            batch_size = args.batch_size)
    valloader = DataLoader(val_dataset,
                        num_workers = args.num_workers,
                        batch_size = args.batch_size) 

    return trainloader, valloader

def train(model, criterion, optimizer, trainloader, valloader, args, device): 
    
    best_vloss = 1_000_000.
    train_loss_history = []
    val_loss_history = []

    for epoch in range(args.epochs):
        # training
        model.train(True)

        tloss_history = []
        tcorrect = 0
        ttotal = 0

        for _, data in enumerate(trainloader):
            img0, img1, _, _, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.float().to(device)
            optimizer.zero_grad()
            
            output = model(img0, img1).squeeze(1)
            loss = criterion(output, label)

            tcorrect += torch.count_nonzero(label == (output > 0.5)).item()
            ttotal += len(label)

            loss.backward()
            optimizer.step()
            tloss_history.append(loss.item())

        # validation
        model.eval()

        vloss_history = []
        vcorrect = 0
        vtotal = 0

        with torch.no_grad():
            for _, vdata in enumerate(valloader):
                vimg0, vimg1, _, _, vlabel = vdata
                vimg0, vimg1, vlabel = vimg0.to(device), vimg1.to(device), vlabel.float().to(device)

                voutput = model(vimg0, vimg1).squeeze(1)
                vloss = criterion(voutput, vlabel)
                
                vcorrect += torch.count_nonzero(vlabel == (voutput > 0.5)).item()
                vtotal += len(vlabel)

                vloss_history.append(vloss.item())

        # Calculate train and val loss 
        avg_tloss = np.mean(tloss_history)
        avg_vloss = np.mean(vloss_history)
        avg_tacc = tcorrect / ttotal
        avg_vacc = vcorrect / vtotal
        
        # Track best performance, and save the model's state
        if epoch > 0 and avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'./output/siamese_model_e{epoch}_b{args.batch_size}_lr{args.lr}_num{args.num_samples}_emb{args.emb_size}.pth'
            torch.save(model.state_dict(), model_path)

        train_loss_history.append(tloss_history)
        val_loss_history.append(vloss_history)

        if (epoch + 1) % 10 == 0:
            folder = f'./output/{datetime.today().strftime("%d%m%Y")}/'
            if not os.path.exists(folder):
                os.mkdir(folder)
            with open(f'{folder}training_loss_e{epoch}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.txt', 'a+') as f:
                f.write(f'[{time.ctime()}] [Epoch {epoch}] train loss: {avg_tloss:.3f} val loss: {avg_vloss:.3f} train acc: {avg_tacc:.3f} val acc: {avg_vacc:.3f}')

            model_path = f'{folder}siamese_model_e{epoch}_b{args.batch_size}_lr{args.lr}_num{args.num_samples}_emb{args.emb_size}.pth'
            torch.save(model.state_dict(), model_path)
            
            with open(f'{folder}train_loss_history_e{epoch}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.pickle', 'wb') as handle:
                pickle.dump(train_loss_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{folder}val_loss_history_e{epoch}_b{args.batch_size}_lr{args.lr}_n{args.num_samples}_emb{args.emb_size}.pickle', 'wb') as handle:
                pickle.dump(val_loss_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test(): 
    pass

def main(): 
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--num-samples', type=int, default=1024, metavar='N',
                        help='size of training samples (default: 1024)')
    parser.add_argument('--emb-size', type=int, default=20, metavar='N',
                        help='size of feature embedding (default: 20)')
    parser.add_argument('--train-val-split', type=float, default=0.8, metavar='LR',
                        help='percentage of train to val in data (default: 0.8)')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                        help='number of workers (default: 1)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')

    # perform the configurations 
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # ensure reproducibility 
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # getting the train and val dataloaders 
    full_dataset = SiameseDataset(
        images_folder_path='./data/images/train/', 
        transform=transforms, 
        num_samples=args.num_samples
    )
    trainloader, valloader = dataloader(full_dataset, args)

    # setting up training parameters 
    model = SiameseModel(emb_size=args.emb_size)
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # model training 
    train(model, criterion, optimizer, trainloader, valloader, args, device)

if __name__ == '__main__':
    main()