import torch.nn as nn


class BaseSequenceGenerator(nn.Module):
    def __init__(self):
        super(BaseSequenceGenerator, self).__init__()

    def generate_dummy_data(self, lr_size):
        """ Generate random input tensors for function `step`
        """
        return None

    def profile(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """ Interface (support DDP)
        """
        pass

    def forward_sequence(self, lr_data):
        """ Forward a whole sequence (for training)
        """
        pass

    def step(self, *args, **kwargs):
        """ Forward a single frame
        """
        pass

    def infer_sequence(self, lr_data, device):
        """ Infer a whole sequence (for inference)
        """
        pass


class BaseSequenceDiscriminator(nn.Module):
    def __init__(self):
        super(BaseSequenceDiscriminator, self).__init__()

    def forward(self, *args, **kwargs):
        """ Interface (support DDP)
        """
        pass

    def step(self, *args, **kwargs):
        """ Forward a singe frame
        """
        pass

    def forward_sequence(self, data, args_dict):
        """ Forward a whole sequence (for training)
        """
        pass
