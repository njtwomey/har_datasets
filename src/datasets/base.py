from os.path import join, basename, splitext

from src import BaseGraph, DatasetMeta


class Dataset(BaseGraph):
    def __init__(self, name, *args, **kwargs):
        super(Dataset, self).__init__(
            name=name,
            parent=None,
            meta=DatasetMeta(name)
        )
    
        zip_name = kwargs.get('unzip_path', lambda x: x)(splitext(basename(
            self.meta.meta['download_urls'][0]
        ))[0])
        self.unzip_path = join(self.meta.zip_path, splitext(zip_name)[0])
    
        # Build the indexes
        self.index.add_output(
            key='label',
            func=self.build_label,
            path=self.unzip_path,
            inv_lookup=self.meta.inv_act_lookup
        )
        
        self.index.add_output(
            key='fold',
            func=self.build_fold,
            path=self.unzip_path
        )
        
        self.index.add_output(
            key='index',
            func=self.build_index,
            path=self.unzip_path
        )
        
        # Build list of outputs
        for location, modalities in self.meta['locations'].items():
            for modality, is_active in modalities.items():
                if is_active:
                    self.outputs.add_output(
                        key=(modality, location),
                        func=self.build_data,
                    )
    
    @property
    def identifier(self):
        return self.name
    
    def build_label(self, *args, **kwargs):
        raise NotImplementedError

    def build_index(self, *args, **kwargs):
        raise NotImplementedError

    def build_fold(self, *args, **kwargs):
        raise NotImplementedError

    def build_data(self, *args, **kwargs):
        raise NotImplementedError
