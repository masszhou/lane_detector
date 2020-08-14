def build_validate_fn(params):
    if params.dataset_name == "tusimple":
        from instances.validator_tusimple import validate_fn
        return validate_fn
    elif params.dataset_name == "collections":
        # ToDo: add culane and bdd100k validation
        from instances.validator_tusimple import validate_fn
        return validate_fn
    else:
        return None
