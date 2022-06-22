import functools
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet


def import_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.

    From: https://stackoverflow.com/a/34963527/9807171
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    try:
        return getattr(module_path, class_name)
    except AttributeError as err:
        raise ImportError('Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)) from err


class ClassHolder:
    def __init__(self, accept_module_str=True):
        self.classes = {}
        self.flexible_import = accept_module_str

    def add_class(self, c, class_name=None):
        key = class_name if class_name else c.__name__
        self.classes[key] = c

    def held(self, c):
        """Decorator function to register a class to an instance of ClassHolder"""
        self.add_class(c)
        return c

    def __getitem__(self, key):
        try:
            return self.classes[key]
        except KeyError as e:
            if self.flexible_import:
                return import_string(key)
            else:
                raise e
    
    def keys(self):
        return self.classes.keys()
    

model_holder = ClassHolder()


model_holder.add_class(RandomForestRegressor)
model_holder.add_class(ElasticNet)
