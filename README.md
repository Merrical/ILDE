# Instance-dependent Label Distribution Estimation for Learning with Label Noise (ILDE)
#### [Project Page](https://github.com/Merrical/ILDE)

This repo contains the official implementation of our paper: Instance-dependent Label Distribution Estimation for Learning with Label Noise (IJCV 2024).
<p align="center"><img src="https://raw.githubusercontent.com/Merrical/ILDE/master/ILDE_overview.png" width="90%"></p>

#### [Paper](https://arxiv.org/pdf/2212.08380)

### Implementation 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ILDE(nn.Module):
    def __init__(self, num_samples, num_classes=10, alpha=3.0, beta=0.7, tau=3.0, delta=1.0):
        super(ILDE, self).__init__()
        self.num_classes = num_classes
        self.target = torch.zeros(num_samples, self.num_classes).cuda()

        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.delta = delta

    def forward(self, index, pred, labels):
        y_pred = F.softmax(pred, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()

        label_one_hot = F.one_hot(labels, self.num_classes).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        ce_loss = (-1 * torch.sum(label_one_hot * torch.log(y_pred), dim=1)).mean()

        label_dist_matrix = F.normalize(torch.mm(y_pred.t(), y_pred), p=1, dim=1)
        L2_LDM_dominant = (label_dist_matrix * label_dist_matrix).sum() - torch.diag(label_dist_matrix * label_dist_matrix).sum()
        label_dist_is = torch.mm(y_pred_, label_dist_matrix.t()).float().detach()
        label_dist_is = torch.softmax(label_dist_is / self.tau, dim=1)
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * label_dist_is
        ld_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()

        loss = ce_loss + self.alpha * ld_reg + self.delta * L2_LDM_dominant
        return loss
```

### Usage

```python
loss_func = ILDE(num_samples=train_set.__len__(), num_classes=num_classes, alpha=alpha, beta=beta, tau=tau, delta=delta)

model.train()
for step, (imgs, labels, index) in enumerate(train_loader):
    outputs = model(imgs.cuda())
    loss = loss_func(index, outputs, labels.cuda())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Hyperparameters for synthetic noisy datasets 

<p align="center"><img src="https://raw.githubusercontent.com/Merrical/ILDE/master/hyper_syn.png" width="50%"></p>

### Bibtex
```
@article{liao2024ILDE,
  title={Instance-dependent Label Distribution Estimation for Learning with Label Noise},
  author={Liao, Zehui and Hu, Shishuai and Xie, Yutong and Xia, Yong},
  journal={International Journal of Computer Vision},
  pages={1--13},
  year={2024},
  publisher={Springer}
}
```

### Contact Us
If you have any questions, please contact us ( merrical@mail.nwpu.edu.cn ).