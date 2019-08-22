from os.path import join, basename, splitext

from src import BaseGraph, DatasetMeta


class Dataset(BaseGraph):
    def __init__(self, name, *args, **kwargs):
        super(Dataset, self).__init__(name=name)
        
        self.meta = DatasetMeta(self.name)
        
        zip_name = kwargs.get('unzip_path', lambda x: x)(splitext(basename(
            self.meta.meta['download_urls'][0]
        ))[0])
        self.unzip_path = join(self.meta.zip_path, splitext(zip_name)[0])
    
    @property
    def identifier(self):
        return self.name
    
    def compose_index(self):
        return self.node(
            node_name=self.build_path('index'),
            func=self.build_index,
            kwargs=dict(
                path=self.unzip_path
            )
        )
    
    def compose_fold(self):
        return self.node(
            node_name=self.build_path('fold'),
            func=self.build_fold,
            kwargs=dict(
                path=self.unzip_path,
            )
        )
    
    def compose_label(self):
        return self.node(
            node_name=self.build_path('label'),
            func=self.build_label,
            kwargs=dict(
                path=self.unzip_path,
                inv_lookup=self.meta.inv_act_lookup
            )
        )
    
    def compose_outputs(self):
        outputs = dict()
        for location, modalities in self.meta.meta['locations'].items():
            for modality, active in modalities.items():
                if not active:
                    continue
                outputs[modality, location] = self.node(
                    node_name=self.build_path(modality, location),
                    func=self.build_data,
                    sources=None,
                    kwargs=dict(
                        path=self.unzip_path,
                        key=(modality, location),
                    )
                )
        return outputs
