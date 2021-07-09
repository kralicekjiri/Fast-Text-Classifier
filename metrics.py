import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score


class Accuracy(object):
    """docstring for PR"""
    def __init__(self):
        super(Accuracy, self).__init__()

        self.accum = {
            "target": [],
            "pred": [],
            "target3": [],
            "pred3": [],
            "target5": [],
            "pred5": [],
        }

    def add(self, target, pred, thresh):
        self.accum["target"].append(((np.sum(target, axis=1) > 0) + 0)[0])
        pred_tr = np.greater(pred[::, 1, ::], thresh)
        self.accum["pred"].append(((np.sum(pred_tr, axis=1) > 0) + 0)[0])

        self.accum["target3"] += target[0][:9].tolist()
        pred_tr3 = np.greater(pred[::, 1, ::], thresh) + 0
        self.accum["pred3"] += pred_tr3[0][:9].tolist()

        self.accum["target5"] += target[0][9:].tolist()
        pred_tr5 = np.greater(pred[::, 1, ::], thresh) + 0
        self.accum["pred5"] += pred_tr5[0][9:].tolist()

    def result(self):
        print("--------------------")
        print("Accuracy results")
        print("--------------------")

        pr = precision_score(self.accum["target"], self.accum["pred"])
        rs = recall_score(self.accum["target"], self.accum["pred"])
        f = f1_score(self.accum["target"], self.accum["pred"])
        print(f"Precision: {pr:.4}")
        print(f"Recall: {rs:.4}")
        print(f"F-score: {f:.4}\n")

        pr3 = precision_score(self.accum["target3"], self.accum["pred3"])
        rs3 = recall_score(self.accum["target3"], self.accum["pred3"])
        f3 = f1_score(self.accum["target3"], self.accum["pred3"])
        print(f"Precision 3x3: {pr3:.4}")
        print(f"Recall 3x3: {rs3:.4}")
        print(f"F-score 3x3: {f3:.4}\n")

        pr5 = precision_score(self.accum["target5"], self.accum["pred5"])
        rs5 = recall_score(self.accum["target5"], self.accum["pred5"])
        f5 = f1_score(self.accum["target5"], self.accum["pred5"])
        print(f"Precision 5x5: {pr5:.4}")
        print(f"Recall 5x5: {rs5:.4}")
        print(f"F-score 5x5: {f5:.4}\n")


class Time(object):
    """docstring for Time"""

    def __init__(self, num_imgs):
        super(Time, self).__init__()
        self.start_time = 0
        self.stop_time = 0
        self.total_time = 0
        self.num_imgs = num_imgs  # number of images processed

    def start(self):
        self.start_time = time.time()

    def stop(self, ):
        self.stop_time = time.time()
        self.total_time += self.stop_time - self.start_time

    def result(self, name):
        print("\n-------------------------------")
        print("%s results" % name)
        print("Total time: %fs" % self.total_time)
        print("Average Inference Time: %f" % (self.total_time / self.num_imgs))
        print("FPS: %f" % (1 / (self.total_time / self.num_imgs)))
        print("-------------------------------\n")
