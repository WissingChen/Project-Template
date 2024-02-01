from .loss import compute_lm_loss, compute_ce_loss
from .lr_scheduler import build_lr_scheduler
from .optimizer import build_optimizer


loss_fns = {"nlg_loss": compute_lm_loss, "ce": compute_ce_loss}
