import yaml


def split_module_and_class(class_path):
    module_name = ".".join(class_path.split(".")[:-1])
    class_name = class_path.split(".")[-1]
    return module_name, class_name


def dynamic_load_class(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def dynamic_load_function(module_name, function_name):
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)


def recursive_load_class(val):

    if isinstance(val, dict):

        for k, v in val.items():
            val[k] = recursive_load_class(v)

        if "class_path" in val:

            if "init_args" not in val:
                return dynamic_load_class(*split_module_and_class(val["class_path"]))()

            return dynamic_load_class(*split_module_and_class(val["class_path"]))(
                **val["init_args"]
            )

    elif isinstance(val, list):

        for i, v in enumerate(val):
            val[i] = recursive_load_class(v)

    return val


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return recursive_load_class(config)
