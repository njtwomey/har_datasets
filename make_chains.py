from src import load_datasets_metadata, load_chains_metadata
from src import chain_importer
from src import dot_env_stuff


def main():
    datasets = load_datasets_metadata()
    chains = load_chains_metadata()
    for name in datasets.keys():
        for chain in reversed(list(chains.keys())):
            print(name, chain)
            chain_importer(
                chain, name=name
            ).evaluate_outputs()


if __name__ == '__main__':
    dot_env_stuff(main)
