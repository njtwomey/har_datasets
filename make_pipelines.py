from src import load_datasets_metadata, load_pipelines_metadata
from src import pipeline_importer
from src import dot_env_decorator
from src import randomised_order


@dot_env_decorator
def main():
    datasets = load_datasets_metadata()
    pipelines = load_pipelines_metadata()
    for name in randomised_order(datasets.keys()):
        for pipeline in randomised_order(pipelines.keys()):
            print(name, pipeline)
            pipeline_importer(pipeline, name=name).evaluate_outputs()


if __name__ == "__main__":
    main()
