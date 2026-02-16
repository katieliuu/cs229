import numpy as np
import csv

def create_labels(csv_path, label_col, header_included):
    with open(csv_path, "r", newline="") as f:
        headers = f.readline().strip().split(",")
    x_cols = [i for i, h in enumerate(headers) if h in header_included]
    l_col = headers.index(label_col)
    X = np.genfromtxt(
        csv_path, delimiter=",", skip_header=1, usecols=x_cols,
        dtype=float, filling_values=np.nan, missing_values=["", "NA", "NaN"]
    )
    y = np.genfromtxt(
        csv_path, delimiter=",", skip_header=1, usecols=l_col,
        dtype=float, filling_values=np.nan, missing_values=["", "NA", "NaN"]
    )

    return X, y

def main():
    header_included = ["SEQN", "LBXSGL", "LBXIN", "LBXGH"]
    label_col = "LBXGLU"
    X, y = create_labels(
        "data/analysis_ready/nhanes_joined_2017_2018.csv",
        label_col=label_col,
        header_included=header_included
    )
    # Drop rows where label is missing
    mask = (~np.isnan(y)) & (~np.isnan(X).any(axis=1))
    X = X[mask]
    y = y[mask]

    print(f'Number of rows with missing labels: {np.sum(np.isnan(y))}')
    print(f'Number of rows with missing features: {np.sum(np.isnan(X).any(axis=1))}')
    print(f'Number of rows with missing labels and features: {np.sum(np.isnan(y) & np.isnan(X).any(axis=1))}')
    print(f'Number of rows with no missing values: {np.sum(~np.isnan(y) & ~np.isnan(X).any(axis=1))}')

    y_bin = (y > 128).astype(int).reshape(-1, 1)
    out_headers = header_included + [f"{label_col}_bin"]
    out = np.hstack([X, y_bin])

    with open("data/analysis_ready/small_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_headers)
        writer.writerows(out)

    print("Small dataset created")


if __name__ == "__main__":
    main()
