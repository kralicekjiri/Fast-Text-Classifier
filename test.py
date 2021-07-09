import argparse
import datetime
import torch
import metrics
from model import FTC
from dataset import TextDisTwitterBlocks
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--model_path', required=True, help="Path to a model.")
    parser.add_argument('--bs', '--batch_size', default=1, type=int, help="batch size", dest='batch_size')
    parser.add_argument('--nw', '--num_workers', default=12, type=int, help="number of workers", dest='num_workers')
    parser.add_argument('--width', type=int, required=False, default=360, help="Input image width.")
    parser.add_argument('--height', type=int, required=False, default=360, help="Input image height.")
    parser.add_argument('--thresh', type=float, required=False, default=0.5, help="Text/non-text decision threshold.")
    parser.add_argument('--dataset_thresh', type=float, required=False, default=0.01,
                        help="Defines a minimum text area coverage of a block to label as text block.")
    parser.add_argument('--cuda', type=bool, required=False, default=False)
    parser.add_argument('--mask_path', type=str, default="./data/Twitter1M/masks/",
                        help="Path to binary text/non masks.")

    opts = parser.parse_args()

    # paths should contain name of dataset -- textdis, twitter to recognize image source

    eval_dataset = TextDisTwitterBlocks(
        file_path=opts.dataset_path,
        mask_path=opts.mask_path,
        size=(opts.height, opts.width),
        thresh=opts.dataset_thresh,
        scales=(3, 5),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers
    )

    print(f"Number of images: {eval_dataset.__len__()}")
    date_file = "./result_analysis_" + str(datetime.date.today()) + ".csv"

    device = "cuda" if opts.cuda else "cpu"

    model = FTC(
        model_path=opts.model_path,
        device=device
    )

    accuracy = metrics.Accuracy()

    time_inf = metrics.Time(eval_dataset.__len__())  # inference time
    time_wall = metrics.Time(eval_dataset.__len__())  # wall time
    time_wall.start()

    with torch.no_grad():
        for i, (batch, target, image_name) in enumerate(tqdm(eval_loader)):

            if opts.cuda:
                target = target.cuda(non_blocking=True)
                batch = batch.cuda(non_blocking=True)

            time_inf.start()
            output = model.inference(batch)
            time_inf.stop()
            # batch, blocks, classes -> batch, classes, blocks
            output = output.permute(0, 2, 1)

            pred = torch.nn.functional.softmax(output, dim=1)
            pred = pred.cpu().numpy()
            pred_max = np.argmax(pred, axis=1)
            target = target.cpu().numpy()

            accuracy.add(target, pred, opts.thresh)

        time_wall.stop()

        time_wall.result("Wall time")
        time_inf.result("Pure inference")

        accuracy.result()
