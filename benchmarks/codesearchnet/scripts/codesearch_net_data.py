import pandas as pd
dataset_path = "benchmarks/codesearchnet/data/code_search_net_train.csv"


def generate_report():
    pd_records = pd.read_csv(dataset_path)
    print(pd_records.shape)


def create_random_record():
    pd_records = pd.read_csv(dataset_path)
    print(pd_records.shape)

    # Get 100 random records
    sample_records = pd_records.sample(n=100000, random_state=42)
    sample_records[['code']].to_csv("benchmarks/codesearchnet/data/csn_1lakh.csv", index=False)
    print(sample_records.head())  # Optional: view the first few rows





if __name__ == "__main__":
    # generate_report()
    create_random_record()
