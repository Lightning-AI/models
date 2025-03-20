import operator

from lightning_utilities import compare_version, module_available

_LIGHTNING_AVAILABLE = module_available("lightning")
_LIGHTNING_GREATER_EQUAL_2_5_1 = compare_version("lightning", operator.ge, "2.5.1")
_PYTORCHLIGHTNING_AVAILABLE = module_available("pytorch_lightning")
_PYTORCHLIGHTNING_GREATER_EQUAL_2_5_1 = compare_version("pytorch_lightning", operator.ge, "2.5.1")
