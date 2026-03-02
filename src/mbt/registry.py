API_REGISTRY = {}
TASK_REGISTRY = {}


def register_api(name):
    def decorator(func):
        assert name not in API_REGISTRY
        API_REGISTRY[name] = func
        return func

    return decorator


def register_task(name):
    def decorator(cls):
        assert name not in TASK_REGISTRY
        TASK_REGISTRY[name] = cls
        return cls

    return decorator
