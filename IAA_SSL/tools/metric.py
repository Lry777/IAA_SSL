import torch
import torch.nn.functional as F
from scipy import stats
import numpy as np




class Accuracy:
    @torch.no_grad()
    def __call__(self, prediction, target):
        # print(prediction.shape, target.shape)
        N = prediction.size(0)
        assert len(prediction) == len(target), 'ce_loss: size woring1'
        no_idx = 0
        for i, tar in enumerate(target):
            if tar < 0:
                prediction = prediction[torch.arange(prediction.size(0)) != i]
                target = target[torch.arange(target.size(0)) != i]
                no_idx +=1
        assert len(prediction) == len(target), 'ce_loss: size woring2'
        # print(torch.mean((torch.argmax(prediction, dim=1) == target).float()).item(), no_idx, N)
        # return (torch.mean((torch.argmax(prediction, dim=1) == target).float()).item() + no_idx) / N

        return torch.mean((torch.argmax(prediction, dim=1) == target).float()).item()


class MAE:
    @torch.no_grad()
    def __call__(self, prediction, target):
        return F.l1_loss(prediction, target).item()


class AccuracyFromDistribution:
    def __init__(self, cut_off=5.0):
        self.cut_off = cut_off

    @staticmethod
    def get_score_from_distribution(distribution: torch.Tensor):
        # distribution: (N, num_classes)
        N, num_classes = distribution.size()
        arrange_index = torch.stack([torch.arange(1, num_classes + 1) for _ in range(N)], dim=0)

        return torch.sum(distribution * arrange_index.float().to(distribution.device), dim=-1)

    @torch.no_grad()
    def __call__(self, prediction, target):
        # print(self.get_score_from_distribution(prediction), self.get_score_from_distribution(target))
        prediction_label = self.get_score_from_distribution(prediction) > self.cut_off
        target_label = self.get_score_from_distribution(target) > self.cut_off
        # print(prediction_label, target_label)
        return torch.mean((prediction_label == target_label).float()).item()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def spearmanr(pre, tar, is_temp = True):
    pre = pre.cpu().detach().numpy()
    tar = tar.cpu().detach().numpy()
    if is_temp:
        temp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    else:
        temp = np.array([1])
    p = pre * temp
    t = tar * temp

    return stats.spearmanr(np.sum(p, axis=1), np.sum(t, axis=1))[0]

def pearsonr(pre, tar, is_temp = True):
    pre = pre.cpu().detach().numpy()
    tar = tar.cpu().detach().numpy()
    if is_temp:
        temp = np.array([1,2,3,4,5,6,7,8,9,10])
    else:
        temp = np.array([1])
    p = pre * temp
    t = tar * temp

    return stats.pearsonr(np.sum(p, axis=1), np.sum(t, axis=1))[0]

if __name__ == '__main__':
    def run_accuracy():
        prediction = torch.randn((1000, 10)).cuda()
        target = torch.randint(0, 10, (1000,)).cuda()
        print(prediction.shape, target.shape)
        print("===> acc: ", Accuracy()(prediction, target))


    def run_mse():
        prediction = torch.randn((2,)).cuda()
        target = torch.randn((2,)).cuda()
        print("==> prediction: ", prediction)
        print("==> target: ", target)
        print("===> mse: ", MAE()(prediction, target))


    def run_acc_from_distribution():
        prediction = torch.softmax(torch.randn(5, 10), dim=-1)
        target = torch.softmax(torch.randn(5, 10), dim=-1)
        print("==> prediction: ", prediction)
        print("==> target: ", target)
        print("===> accuracy from distribution: ", AccuracyFromDistribution()(prediction, target))

    def run_sm():
        prediction = torch.softmax(torch.randn(5, 10), dim=-1)
        target = torch.softmax(torch.randn(5, 10), dim=-1)
        print("==> prediction: ", prediction.shape)
        print("==> target: ", target.shape)
        # sm = stats.spearmanr(prediction[0], target[0])
        print(spearmanr(prediction, target))

    def run_ps():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prediction = torch.softmax(torch.randn(5, 10), dim=-1)
        target = torch.softmax(torch.randn(5, 10), dim=-1)
        print("==> prediction: ", prediction.shape)
        print("==> target: ", target.shape)

        print(pearsonr(prediction, target))

    # run_accuracy()
    # run_mse()
    # run_acc_from_distribution()
    run_sm()
    # run_ps()

