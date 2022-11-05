import torch
import torch.nn as nn
from .resnet import resnet18
import numpy as np

class SkillClassifier(nn.Module):
    def __init__(self, num_cls, num_in_c):
        super().__init__()
        self.classifier = resnet18(num_classes=num_cls, num_in_c=num_in_c)

    def forward(self, x):
        return self.classifier(x) # (b, 2)

    def get_loss(self, x, y, w):
        '''

        :param x: (b, 2, h, w)
        :param y: (b, 2)
        :return:
        '''
        pred = self.classifier(x) # (b, 2)
        criterion = nn.BCEWithLogitsLoss(reduce=False)

        loss = criterion(pred, y)
        assert loss.shape == w.shape
        f_loss = torch.mean(loss * w) * 2.
        return f_loss

if __name__ == '__main__':

    model = SkillClassifier(num_cls=2, num_in_c=2)
    data = torch.zeros((3, 2, 32, 32))

    label = np.array([[0, 1], [1, 0], [1, 1]]).astype("float32")
    label = torch.from_numpy(label)
    pred = model(data)

    pos_w = np.array([[0, 1], [0, 1], [0, 1]])
    pos_w = torch.from_numpy(pos_w)
    loss = model.get_loss(data, label, pos_w)
    print(pred)
    print(loss)




