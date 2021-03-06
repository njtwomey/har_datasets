from src import load_datasets_metadata
from src import load_pipelines_metadata
from src import pipeline_importer
from src import randomised_order


def main():
    datasets = load_datasets_metadata()
    pipelines = load_pipelines_metadata()
    for name in randomised_order(datasets.keys()):
        for pipeline in randomised_order(pipelines.keys()):
            print(name, pipeline)
            pipeline_importer(pipeline, name=name).evaluate_outputs()


if __name__ == "__main__":
    main()
