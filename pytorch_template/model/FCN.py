from pytorch_template.torch_nn_module import torch_nn_module
import itertools
import torch


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


@torch_nn_module
class FCN:
    input_dim: int
    output_dim: int
    fc_dims: list[int]
    activation = torch.nn.functional.relu

    def __post_init__(self):
        self.fc_layers = [torch.nn.Linear(_in, _out)
                          for _in, _out in pairwise(itertools.chain([self.input_dim], self.fc_dims))]

    def forward(self, input_tensor):
        out = input_tensor
        for layer, activation_fn in zip(self.fc_layers[:-1], itertools.repeat(self.activation)):
            out = layer(out)
            out = activation_fn(out)
        out = self.fc_layers[-1](out)
        return out

