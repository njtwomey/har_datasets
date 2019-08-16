from src import BaseDataset


class StatisticalFeatures(BaseDataset):
    def __init__(self, name, parent):
        super(StatisticalFeatures, self).__init__(name)
        self.parent = parent
    
    def compose(self):
        pass