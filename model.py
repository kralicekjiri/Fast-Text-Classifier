import torch
import torch.backends.cudnn as cudnn
import pdb
from ftc import ftc


class FTC(object):
    """docstring for FTC"""

    def __init__(self,
                 device="cuda",
                 model_path=None,
                 dropout=False,
                 ):
        super(FTC, self).__init__()
        self.type = type
        self.device = device
        self.model_path = model_path

        self.model = ftc(
            # num_classes=2,
            dropout=dropout,
        )

        if self.model_path is not None:
            print("Loading FTC model. Loading weights from " + self.model_path)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

        if self.device is "cuda":
            self.model.to(device)
            cudnn.benchmark = True

    def rawNet(self):
        # used for training out of class
        return self.model

    def inference(self, batch):

        self.model.eval()

        if self.device == "cuda":
            batch = batch.cuda(non_blocking=True)
        else:
            self.model.to("cpu")

        output = self.model(batch)
        return output

        # pred = torch.nn.functional.softmax(output, dim=1)
        # pred = pred.cpu().numpy()
        # pred_max = np.argmax(pred, axis=1)
        #
        # return pred, pred_max
