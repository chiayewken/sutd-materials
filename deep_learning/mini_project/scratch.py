from pascal_classify import HyperParams


class MyParams(HyperParams):
    def __init__(self, one=1, two=2, **kwargs):
        self.one = one
        self.two = two

        kwargs.update(vars(self))
        super().__init__(**kwargs)


a = MyParams(learn_rate=10)


def fn(**kwargs):
    print(kwargs)


fn(**dict(one=1, two=2))
