# BaseAssigner class for target assign


class BaseAssigner:
    def __init__(self):
        super(BaseAssigner, self).__init__()

    def assign_targets(self):
        raise NotImplementedError
