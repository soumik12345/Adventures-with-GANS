from torch.optim import Adam


class AdamOptimizer:

    def __init__(self, model, learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999):
        self.model = model
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    def get_optimizer(self):
        self.optimizer = Adam(
            self.model.parameters(),
            lr = self.learning_rate,
            betas = (
                self.beta_1,
                self.beta_2
            )
        )
        return self.optimizer