import torch
import torch.nn as nn
import torch.nn.functional as F


class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_estimate: torch.Tensor, p_target: torch.Tensor):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()

class emd_eval(nn.Module):
    def __init__(self):
        super(emd_eval, self).__init__()

    def forward(self, p_estimate: torch.Tensor, p_target: torch.Tensor):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.mean(torch.pow(torch.abs(cdf_diff), 1))
        return samplewise_emd.mean()

def single_emd_loss(p, q, r=2):
    """
    Implementation from https://github.com/kentsyx/Neural-IMage-Assessment
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def emd_loss(p, q, r=2):
    """
    Implementation from https://github.com/kentsyx/Neural-IMage-Assessment
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size


def earth_mover_distance(input: torch.Tensor, target: torch.Tensor, r: float = 2):
    """
    Batch Earth Mover's Distance implementation.
    Args:
        input: B x num_classes
        target: B x num_classes
        r: float, to penalize the Euclidean distance between the CDFs

    Returns:

    """
    N, num_classes = input.size()
    input_cumsum = torch.cumsum(input, dim=-1)
    target_cumsum = torch.cumsum(target, dim=-1)

    diff = torch.abs(input_cumsum - target_cumsum) ** r

    class_wise = (torch.sum(diff, dim=-1) / num_classes) ** (1. / r)
    scalar_ret = torch.sum(class_wise) / N
    return scalar_ret

class EarthMoverDistanceLoss(nn.Module):
    def __init__(self, r: float = 2.0):
        super().__init__()
        self.r = r

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return earth_mover_distance(input, target, r=self.r)


def ce_loss(logits, targets, device, use_hard_labels=True, reduction='none', softmax = nn.Softmax( dim=-1)):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        N, num_classes = logits.size()
        # ce_loss = nn.CrossEntropyLoss()

        assert len(targets) == len(logits), 'ce_loss: size woring1'
        weight = torch.Tensor([1]*N)
        n = 0
        # i = 0
        # while i<len(targets):
        #     if targets[i]<0:
        #         targets = targets[torch.arange(targets.size(0))!=i]
        #         logits = logits[torch.arange(logits.size(0))!=i]
        #     else:
        #         i +=1
        #         n +=1

        for i, tar in enumerate(targets):
            if tar<0:
                weight[i] = 0
                targets[i] = 0
            else:
                n +=1
        # print(n)
        # print(len(targets), len(logits))
        # log_pred = F.log_softmax(logits, dim=-1)
        # N, num_classes = logits.size()
        log_pred = softmax(logits)

        loss = F.nll_loss(log_pred, targets, reduction=reduction)
        loss = loss * weight.to(device)
        # print(loss.shape)
        if N ==0:
            return 0
        else:
            return torch.sum(loss) / N
        # return torch.sum(loss) / n
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

if __name__ == '__main__':
    def run_earth_mover_distance():
        a = torch.softmax(torch.randn((5, 100)), dim=-1)
        b = torch.softmax(torch.randn((5, 100)), dim=-1)
        ret_emd = emd_loss(a.clone(), b.clone())
        ret = earth_mover_distance(a.clone(), b.clone())
        print("===> ret for ori implementation: ", ret_emd)
        print("===> ret for cumsum implementation: ", ret)


    # run_earth_mover_distance()
    # input = torch.softmax(torch.randn((128, 14)), dim=-1)
    # label = torch.randint(low=1, high=14, size=[128])
    # print(input.shape, label.shape)
    # loss = ce_loss(input, label, use_hard_labels=True)
    # print(loss)
    p_target = torch.Tensor([[1, -2, 3, 4], [1, -2, 3, 4]])
    cdf_target = torch.cumsum(p_target, dim=1)
    print(cdf_target)