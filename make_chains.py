from src import load_datasets_metadata, load_chains_metadata
from src import chain_importer
from src import dot_env_stuff
from src import randomised_order


def main():
    datasets = load_datasets_metadata()
    chains = load_chains_metadata()
    for name in randomised_order(datasets.keys()):
        for chain in randomised_order(chains.keys()):
            print(name, chain)
            chain_importer(
                chain, name=name
            ).evaluate_outputs()


if __name__ == '__main__':
    dot_env_stuff(main)
