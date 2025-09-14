import copy
import deepspeed


def prepare_deepspeed(model, deepspeed_plugin):
    """
        Prepares deepspeed model.
        Adapted from TOFU repository, which adapted this code from accelerate:
        https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    """
    config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    config_kwargs["optimizer"] = {"type": None}
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    # set the gradients to false for every parameter
    for param in model.parameters():
        param.requires_grad = False

    return model
