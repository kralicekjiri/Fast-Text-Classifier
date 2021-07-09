import os
import sys
import argparse
import metrics
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from model import FTC
from dataset import TextDisTwitterBlocks

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', required=False,
                        default="/gfs/datasets/twitterStream/twitterDataset240p_100k/train.txt")
    parser.add_argument('--dataset_test', required=False,
                        default="/gfs/datasets/twitterStream/twitterDataset240p_100k/test.txt")
    parser.add_argument('--save_path', required=False, default="./saved_models/FTC/")
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--bs', '--batch_size', default=16, type=int, help="batch size", dest='batch_size')
    parser.add_argument('--nw', '--num_workers', default=12, type=int, help="number of workers", dest='num_workers')
    parser.add_argument('--width', type=int, required=False, default=360)
    parser.add_argument('--height', type=int, required=False, default=360)
    parser.add_argument('--dataset_thresh', type=float, required=False, default=0.01)
    parser.add_argument('--thresh', type=float, required=False, default=0.5)
    parser.add_argument('--mask_path', type=str, default="./data/Twitter1M/masks/",
                        help="Path to binary text/non masks.")

    opts = parser.parse_args()

    if os.path.isdir(opts.save_path):
        raise Exception('Save dir already exists!')
    else:
        os.mkdir(opts.save_path)

    train_dataset = TextDisTwitterBlocks(file_path=opts.dataset_train,
                                         mask_path=opts.mask_path,
                                         size=(opts.height, opts.width),
                                         thresh=opts.dataset_thresh,
                                         scales=(3, 5),
                                         )

    test_dataset = TextDisTwitterBlocks(file_path=opts.dataset_test,
                                        mask_path=opts.mask_path,
                                        size=(opts.height, opts.width),
                                        thresh=opts.dataset_thresh,
                                        scales=(3, 5),
                                        )

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=2*opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    net = FTC()
    model = net.rawNet()

    # weight for merged TextDis and Twitter1M
    weights = torch.tensor([1., 2.19], device="cuda")
    print("weights")
    print(weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(opts.epochs):
        # TRAINING
        loss_sum = 0
        print(f"--------------------\nepoch: {epoch}")

        model.train()
        for i, (batch, target, _) in enumerate(tqdm(train_loader)):

            batch = batch.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(batch)

            # batch, blocks, classes -> batch, classes, blocks
            output = output.permute(0, 2, 1)

            loss = criterion(output, target)
            loss_sum += loss.item()

            # compute gradient and make a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save model every epoch
        print("Saving model...")
        save_path = f"{opts.save_path}FTC_{str(epoch)}.pth"
        torch.save(model.state_dict(), save_path)

        ##
        # TESTING
        print("Testing...")
        model.eval()
        accuracy = metrics.Accuracy()

        with torch.no_grad():
            for i, (batch, target, _) in enumerate(test_loader):
                target = target.cuda(non_blocking=True)
                batch = batch.cuda(non_blocking=True)

                output = model(batch)

                # batch, blocks, classes -> batch, classes, blocks
                output = output.permute(0, 2, 1)
                pred = torch.nn.functional.softmax(output, dim=1)
                pred = pred.cpu().numpy()
                pred_max = np.argmax(pred, axis=1)
                target = target.cpu().numpy()

                accuracy.add(target, pred, opts.thresh)

        accuracy.result()

        ## update LR by gamma every step_size (default 30)
        scheduler.step()

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Training finished")
