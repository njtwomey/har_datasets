from src import load_datasets_metadata, load_pipelines_metadata, load_models_metadata
from src import pipeline_importer, model_importer
from src import dot_env_decorator
from src import randomised_order


@dot_env_decorator
def main():
    datasets = load_datasets_metadata()
    pipelines = load_pipelines_metadata()
    models = load_models_metadata()
    
    for dataset_name in randomised_order(datasets.keys()):
        for pipeline_name in randomised_order(pipelines.keys()):
            pipeline = pipeline_importer(pipeline_name, name=dataset_name)
            for model_name in randomised_order(models.keys()):
                print(dataset_name, pipeline_name, model_name)
                model_importer(
                    model_name, parent=pipeline
                ).evaluate_outputs()


if __name__ == '__main__':
    main()
