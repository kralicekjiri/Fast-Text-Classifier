import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TextDisTwitterBlocks(Dataset):
    def __init__(self,
                 file_path,
                 mask_path,
                 size=(360, 360),
                 thresh=0.01,
                 scales=(3, 5),
                 mask_models=('CRAFT', 'MaskTextSpotter', 'TextSnake'),
                 ):

        with open(file_path, "r") as f:
            lines = f.readlines()

        self.images = [line.replace("\n", "") for line in lines]

        self.size = size
        self.thresh = thresh
        self.scales = scales
        self.mask_models = mask_models
        self.mask_path = mask_path

        # for Twitter1M+TextDis dataset
        mean = (0.543, 0.505, 0.491)
        std = (0.335, 0.328, 0.331)

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_orig = Image.open(self.images[idx]).convert('RGB')
        img = self.transform(img_orig)

        if "textDis" in self.images[idx]:
            gt_path = self.images[idx].replace(".jpg", "_gt.txt").replace("/text/", "/text_gt/")

            if os.path.exists(gt_path):
                with open(gt_path, "r") as file:
                    boxes = file.readlines()

                boxes = [box.replace("\n", "").split(" ") for box in boxes]

                text_boxes = []
                for box in boxes:
                    text_box = np.array([
                        [int(box[0]), int(box[1])],
                        [int(box[2]), int(box[3])],
                        [int(box[4]), int(box[5])],
                        [int(box[6]), int(box[7])]
                    ])

                    text_boxes.append(text_box)

                text_map = self._createMap(img_orig, text_boxes)
                labels = self._blockAnnotation(text_map, self.scales, img_orig.size[0], img_orig.size[1], thresh=self.thresh)

            else:
                num = 0
                for scale in self.scales:
                    num += scale * scale
                labels = torch.zeros(num, dtype=torch.long)

        elif "twitter" in self.images[idx]:
            mask_concat = None

            for model in self.mask_models:
                mask_file = self.mask_path + model + "/" + os.path.basename(self.images[idx])
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                if mask_concat is None:
                    mask_concat = np.copy(mask)
                else:
                    mask_concat += mask

            labels = self._blockAnnotation(mask_concat, self.scales, img_orig.size[0], img_orig.size[1], thresh=self.thresh)

        return img, labels, self.images[idx]

    def _createMap(self, image, polygons):
        width = image.size[0]
        height = image.size[1]

        detection_map = np.zeros((height, width, 1), np.uint8)

        for polygon in polygons:
            polygon = polygon.astype(int)
            pts = np.array(polygon).reshape((-1, 1, 2))
            cv2.fillConvexPoly(detection_map, pts, (1))

        return detection_map

    def _blockAnnotation(self, text_map, scales, width, height, thresh):
        block_labels = np.array([], dtype=int)
        for scale in scales:
            w_window = -(width // -scale)  # ceiling
            h_window = -(height // -scale)  # ceiling
            w_diff = w_window * scale - width
            h_diff = h_window * scale - height
            # minimum text area to be classified as text block
            minimum = h_window * w_window * thresh
            # minimum = 0  # one pixel is a minimum

            for i in range(scale):
                w_start = ((i * w_window) - 1) if i > 0 else 0
                w_end = w_start + w_window - 1
                if w_diff > 0 and i > 0:
                    w_start -= 1
                    w_diff -= 1

                for j in range(scale):
                    h_start = ((j * h_window) - 1) if j > 0 else 0
                    h_end = h_start + h_window - 1
                    if h_diff > 0 and j > 0:
                        h_start -= 1
                        h_diff -= 1

                    # slice of tensor according to the scale
                    block = text_map[h_start:h_end, w_start:w_end]
                    if np.sum(block) > minimum:
                        block_labels = np.append(block_labels, [1])
                    else:
                        block_labels = np.append(block_labels, [0])

        labels = torch.from_numpy(block_labels)

        return labels

