from dataclasses import dataclass, asdict
import functools
import pickle
from typing import OrderedDict
import torch
import gin


def bytes_to_tensor(b) -> torch.Tensor:
    return torch.tensor([int(_b) for _b in b], dtype=torch.uint8)


def load_torch_nn_module(state_dict: OrderedDict, cls=None, init_args:dict=dict()):
    """
    reconstructs a network that had been decorated with torch_nn_module
    :param state_dict:
    :type state_dict:
    :param cls:
    :type cls:
    :return:
    :rtype:
    """
    cls = cls or pickle.loads(bytes(state_dict["_class_name"]))
    init_args = init_args or pickle.loads(bytes(state_dict["_init_args"]))
    instance = cls(**init_args)
    instance.load_state_dict(state_dict)
    return instance


def torch_nn_module(base_cls, nn_module=torch.nn.Module, use_gin:bool=True):
    """
    a decorator that combines Python dataclass and torch.nn.Module, and can optionally be made gin configurable
    Stores some extra information to make reconstruction easier
    :param base_cls: the base class to extend. Should be laid out like a data class, and should define a __post_init__()
    which constructs the layers
    :param use_gin: make the returned class gin configurable if True
    :return: the decorated class
    """
    base = dataclass(base_cls)

    @functools.wraps(base_cls, updated=())
    class out(base, nn_module):
        def __post_init__(self):
            init_args_t = bytes_to_tensor(pickle.dumps(asdict(self)))
            class_name_t = bytes_to_tensor(pickle.dumps(self.__class__, fix_imports=True))
            nn_module.__init__(self)
            self.register_parameter("_init_args", torch.nn.parameter.Parameter(init_args_t, requires_grad=False))
            self.register_parameter("_class_name", torch.nn.parameter.Parameter(class_name_t, requires_grad=False))
            base.__post_init__(self)
            return self

    if use_gin:
        out = gin.configurable(out)
    return out
