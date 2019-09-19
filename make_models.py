from src import load_datasets_metadata, load_pipelines_metadata, load_models_metadata
from src import pipeline_importer, model_importer
from src import dot_env_stuff
from src import randomised_order


def main():
    datasets = load_datasets_metadata()
    chains = load_pipelines_metadata()
    models = load_models_metadata()
    
    for dataset_name in randomised_order(datasets.keys()):
        if dataset_name != 'anguita2013':
            continue
        for chain_name in randomised_order(chains.keys()):
            chain = pipeline_importer(chain_name, name=dataset_name)
            for model_name in randomised_order(models.keys()):
                print(dataset_name, chain_name, model_name)
                model_importer(
                    model_name, parent=chain
                ).evaluate_outputs()


if __name__ == '__main__':
    dot_env_stuff(main)
