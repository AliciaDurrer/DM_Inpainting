""" Script to create train, validation and test data lists with paths to healthy and diseased images. """

from pathlib import Path

import pandas as pd


def create_datalist(sub_dirs):
    data_list = []
    for sub_dir in sub_dirs:
        diseased_paths = sorted(list(sub_dir.glob("**/*t1n-voided.nii.gz")))
        for diseased_path in diseased_paths:
            mask_path = diseased_path.parent / (
                diseased_path.name.replace("t1n-voided.nii.gz", "mask.nii.gz")
            )
            data_list.append({"voided": str(diseased_path), "mask": str(mask_path)})

    return pd.DataFrame(data_list)


def main():
    output_dir = Path("./ids/")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(
        "/home/user/testing"
    )
    test_sub_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    print("test_sub_dirs", test_sub_dirs, len(test_sub_dirs))

    data_df = create_datalist(test_sub_dirs)
    data_df.to_csv(output_dir / "test_data.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
