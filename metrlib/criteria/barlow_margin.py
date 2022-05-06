from .barlow import Criterion as BarlowCriterion
from .margin import Criterion as MarginCriterion

from .margin import ALLOWED_MINING_OPS
from .margin import REQUIRES_BATCHMINER
from .margin import REQUIRES_OPTIM


class Criterion(MarginCriterion):
    def __init__(self, opt, batchminer):
        """
        Args:
        """
        super(Criterion, self).__init__(opt, batchminer)

        self.name           = 'barlow_margin'

        self.alpha = opt.loss_barlow_margin_alpha
        self.barlow = BarlowCriterion(opt)


    def forward(self, batch, labels, **kwargs):
        barlow_loss = self.barlow(batch, labels, **kwargs)
        margin_loss = super(Criterion, self).forward(batch, labels, **kwargs)
        loss = (1 - self.alpha) * margin_loss + self.alpha * barlow_loss

        return loss
