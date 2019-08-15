from os.path import join, splitext, basename

from src import BaseProcessor, dot_env_stuff
from src.utils import unzip_data


class DatasetProcessor(BaseProcessor):
    def __init__(self, name, *args, **kwargs):
        super(DatasetProcessor, self).__init__(name=name)
        
        self.labels = None
        self.fold = None
        self.index = None
        self.outputs = None
        
        zip_name = basename(self.dataset.meta['download_urls'][0])
        unzip_path = join(self.dataset.zip_path, splitext(zip_name)[0])
        
        self.unzip_path = kwargs.get('unzip_path', lambda x: x)(unzip_path)

        self.compose()
        
        self.node(
            node_name='unzipped',
            func=unzip_data,
            kwargs=dict(
                zip_path=self.dataset.zip_path,
                in_name=zip_name,
                out_name=self.unzip_path
            ),
            backend='none'
        )
    
    def build_labels(self, path, *args, **kwargs):
        raise NotImplementedError
    
    def build_folds(self, path, *args, **kwargs):
        raise NotImplementedError
    
    def build_index(self, path, *args, **kwargs):
        raise NotImplementedError
    
    def build_source(self, path, modality, location, *args, **kwargs):
        raise NotImplementedError
    
    def compose(self):
        self.labels = self.node(
            node_name=join(self.dataset.raw_path, 'labels'),
            func=self.build_labels,
            kwargs=dict(
                path=self.unzip_path,
                inv_lookup=self.dataset.inv_lookup
            )
        )
        
        self.fold = self.node(
            node_name=join(self.dataset.raw_path, 'fold'),
            func=self.build_folds,
            kwargs=dict(
                path=self.unzip_path,
            )
        )
        
        self.index = self.node(
            node_name=join(self.dataset.raw_path, 'index'),
            func=self.build_index,
            kwargs=dict(
                path=self.unzip_path
            )
        )
        
        self.outputs = dict(
            accel=dict(),
            gyro=dict(),
            magnet=dict()
        )
        
        for location, amga in self.dataset.meta['locations'].items():
            for acc_mag_gyr, active in amga.items():
                if not active:
                    continue
                self.outputs[acc_mag_gyr][location] = self.node(
                    node_name=join(self.dataset.raw_path, f'{acc_mag_gyr}-{location}'),
                    func=self.build_source,
                    sources=None,
                    kwargs=dict(
                        path=self.unzip_path,
                        modality=acc_mag_gyr,
                        location=location,
                    )
                )
