from typing import Dict, List

import torch.nn as nn


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


_LAYER_TYPE = nn.Module


def model_named_layers(model: nn.Module) -> Dict[str, List[_LAYER_TYPE]]:
    named_layers = {}

    def _falatten_model(model, prefix="model"):
        for module_name in model._modules:
            if len(model._modules[module_name]._modules) > 0:
                _falatten_model(model._modules[module_name], prefix + "." + module_name)
            else:
                current_layer_name = prefix + "." + module_name
                current_layer = getattr(model, module_name)
                named_layers[current_layer_name] = current_layer

    _falatten_model(model)
    return named_layers
