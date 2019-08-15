from src import BaseProcessor


class StatisticalFeatures(BaseProcessor):
    def __init__(self, name, parent):
        super(StatisticalFeatures, self).__init__(name)
        self.parent = parent
    
    def compose(self):
        pass