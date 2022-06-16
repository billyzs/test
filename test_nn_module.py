from pytorch_template.torch_nn_module import load_torch_nn_module
from pytorch_template.model.FCN import FCN
import gin


if __name__ == "__main__":
    import sys
    gin_config_files = sys.argv[1:]
    gin.parse_config_files_and_bindings(gin_config_files, "", skip_unknown=True, print_includes_and_imports=True)
    n = FCN()
    d = n.state_dict()
    nn = load_torch_nn_module(d)
