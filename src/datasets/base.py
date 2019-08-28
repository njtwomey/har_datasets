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
        
        # Build list of outputs
        for location, modalities in self.meta['locations'].items():
            for modality, is_active in modalities.items():
                if is_active:
                    self.add_output(None, (modality, location), self.build_data)
        self.add_extra_kwargs(path=self.unzip_path)
    
    @property
    def identifier(self):
        return self.name
    
    def compose_meta(self, name):
        return self.node(
            node_name=self.build_path(name),
            func=dict(
                index=self.build_index,
                fold=self.build_fold,
                label=self.build_label
            )[name],
            kwargs=dict(
                inv_lookup=self.meta.inv_act_lookup,
                **(self.extra_args or dict())
            )
        )
    
    def make_node(self, in_key, out_key, func):
        return self.node(
            node_name=self.build_path(*out_key),
            func=func,
            sources=None,
            kwargs=dict(
                key=out_key,
                **(self.extra_args or dict())
            )
        )
