from bisect import bisect

class WarmupLR:

    def __init__(self, warmup_steps, warmup_factor, lr_milestones, lr_gamma):
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma
    
    def __call__(self, current_step):
        if current_step <= self.warmup_steps:
            alpha = current_step / float(self.warmup_steps)
            return self.warmup_factor * (1.0 - alpha) + alpha
        else:
            idx = bisect(self.lr_milestones, current_step)
            return pow(self.lr_gamma, idx)
