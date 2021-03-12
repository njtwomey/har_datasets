# -*- coding: utf-8 -*-
import os

from src import DatasetMeta
from src import load_datasets_metadata
from src.utils.loaders import build_path
from src.utils.loaders import load_metadata


def make_links(links, desc="Link"):
    return ", ".join("[{} {}]({})".format(desc, ii, url) for ii, url in enumerate(links, start=1))


def make_dataset_row(dataset):
    # modalities = sorted(set([mn for ln, lm in self.meta['locations'].items() for mn, mv in lm.items() if mv]))

    data = [
        dataset.meta["author"],
        dataset.meta["paper_name"],
        dataset.name,
        make_links(links=dataset.meta["description_urls"], desc="Link"),
        dataset.meta.get("missing", ""),
        make_links(links=dataset.meta["paper_urls"], desc="Link"),
        dataset.meta["year"],
        dataset.meta["fs"],
        ", ".join(dataset.meta["locations"].keys()),
        ", ".join(dataset.meta["modalities"]),
        dataset.meta["num_subjects"],
        dataset.meta["num_activities"],
        ", ".join(dataset.meta["activities"].keys()),
    ]

    return (
        (
            f"| First Author | Paper Name | Dataset Name | Description | Missing data "
            f"| Download Links | Year | Sampling Rate | Device Locations | Device Modalities "
            f"| Num Subjects | Num Activities | Activities | "
        ),
        "| {} |".format(" | ".join(["-----"] * len(data))),
        "| {} |".format(" | ".join(map(str, data))),
    )


def main():
    # Ensure the paths exist
    root = build_path("tables")
    if not os.path.exists(root):
        os.makedirs(root)

    # Current list of datasets
    lines = []
    datasets = load_datasets_metadata()
    for dataset in datasets:
        dataset = DatasetMeta(dataset)
        head, space, line = make_dataset_row(dataset)
        lines.append(line)
    with open(build_path("tables", "datasets.md"), "w") as fil:
        fil.write("{}\n".format(head))
        fil.write("{}\n".format(space))
        for line in lines:
            fil.write("{}\n".format(line))

    # Iterate over the other data tables
    dims = [
        "activities",
        "features",
        "locations",
        "models",
        "pipelines",
        "transformers",
        "visualisations",
    ]

    for dim in dims:
        with open(build_path("tables", f"{dim}.md"), "w") as fil:
            data = load_metadata(f"{dim}.yaml")
            fil.write(f"| Index | {dim[0].upper()}{dim[1:].lower()} | value | \n")
            fil.write(f"| ----- | ----- | ----- | \n")
            if isinstance(data, dict):
                for ki, (key, value) in enumerate(data.items()):
                    if isinstance(value, dict) and "description" in value:
                        value = make_links(value["description"])
                    fil.write(f"| {ki} | {key} | {value} | \n")


if __name__ == "__main__":
    main()
