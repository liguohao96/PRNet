class Configuration(object):
    '''
    store configurations into k-v pairs
    '''
    @classmethod
    def from_dict(cls, dict_obj):
        config = cls()
        for k in dict_obj:
            if isinstance(dict_obj[k], dict):
                if not hasattr(config, k):
                    config[k] = cls()
                config[k] += cls.from_dict(dict_obj[k])
            else:
                config[k] = dict_obj[k]
        return config

    def __init__(self):
        self._container = dict()

    def __getattribute__(self, name):
        if name == '__class__':
            # prevent infinite recursive, while using self.__class__
            return Configuration
        container = super(self.__class__, self).__getattribute__('_container')

        if name in container:
            return container[name]
        else:
            return super(self.__class__, self).__getattribute__(name)

    def __setitem__(self, name, value):
        # conf_obj[conf_key] = value
        container = super(self.__class__, self).__getattribute__('_container')
        container[name] = value
        return value

    def __getitem__(self, name):
        # value = conf_obj[conf_key]
        return self.__getattribute__(name)

    def __iter__(self):
        # only return keys
        container = super(self.__class__, self).__getattribute__('_container')
        ret = list(container).__iter__()
        return ret

    def __add__(self, other):
        new_config = self.__class__()
        for k in self:
            new_config[k] = self[k]
        for k in other:
            new_config[k] = other[k]
        return new_config

    def __iadd__(self, other):
        # make a += b faster
        for k in other:
            self[k] = other[k]
        return self

    def __str__(self):
        lines = []

        container = super(self.__class__, self).__getattribute__('_container')
        for k in container:
            item = self[k]
            if isinstance(item, self.__class__):
                # get lines
                temp_str = str(item)
                # spit into str list
                temp_str = temp_str.split('\n')
                # append k. before the original name
                temp_str = ['{}.{}'.format(k, s.strip()) for s in temp_str]
                lines.extend(temp_str)
            else:
                # add k before value
                lines.append('{}={}'.format(k, item))
        # get the number of characters of the longest str before '='
        max_character = max([len(s.split('=')[0]) for s in lines])
        # get the format str to make output harmony
        format_str = '{{:<{}}}= {{}}'.format(max_character+1)
        # format
        lines = [format_str.format(*[ele.strip()
                                     for ele in s.split('=')]) for s in lines]
        # sort by dictionary order
        lines.sort()
        # concate into a single str
        return '\n'.join(lines)

