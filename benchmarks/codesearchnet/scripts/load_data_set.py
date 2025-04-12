from datasets import load_dataset
import pandas as pd


def load_and_save_dataset_csv():
    ds = load_dataset("Nan-Do/code-search-net-python")
    for split in ds.keys():  # Iterate over dataset splits (train, validation, etc.)
        df = pd.DataFrame(ds[split])
        df.to_csv(f"code_search_net_{split}.csv", index=False)


if __name__ == "__main__":
    load_and_save_dataset_csv()
