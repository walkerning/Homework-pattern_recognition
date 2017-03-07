# -*- coding: utf-8 -*-

class Registry(type):
    _REGISTRY = {}

    def __init__(cls, name, bases, attrs):
        registry_name = getattr(cls, "REGISTRY_NAME", None)
        if registry_name is None:
            return None
        if registry_name not in Registry._REGISTRY:
            Registry._REGISTRY[registry_name] = (cls, {})
        typ = attrs.get("TYPE", None)
        if typ is not None:
            assert typ not in Registry._REGISTRY[registry_name][1], "Error: A {} class of type `{}` is already registered.".format(registry_name, typ)
            Registry._REGISTRY[registry_name][1][typ] = cls

    def populate_all_types(cls):
        return cls.list_registry()

    @classmethod
    def _get_registry(mcls, name, typ):
        return Registry._REGISTRY[name][typ]

    def get_registry(cls, typ):
        return Registry._REGISTRY[cls.REGISTRY_NAME][1][typ]

    def list_registry(cls):
        return Registry._REGISTRY[cls.REGISTRY_NAME][1].keys()
