from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff

folder = Path(__file__).parent
figures = folder / "figures"
figures.mkdir(exist_ok=True)

bands = pd.DataFrame(arff.loadarff(folder / "datasets/cylinder-bands/phpAz9Len.arff")[0])
pages = pd.DataFrame(arff.loadarff(folder / "datasets/page_blocks/dataset_30_page-blocks.arff")[0])

bands = bands.replace([b"?", "?"], float("nan"))
pages = pages.replace([b"?", "?"], float("nan"))

datasets = {
    "bands": (bands, "band_type"),
    "pages": (pages, "class")
}


for name, df, target in [
    ("Cylinder Bands", bands, "band_type"),
    ("Page Blocks", pages, "class")
]:
    print(f"\n===== {name} =====")

    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nFeature types:")
    print(df.dtypes.value_counts())

    print("\nClass distribution:")
    print(df[target].value_counts())
    print(df[target].value_counts(normalize=True) * 100)

    print("\nMissing values:")
    print(df.isna().sum()[df.isna().sum() > 0])

    print("\nNumerical statistics:")
    print(df.describe().round(2))

for name, (df, target) in datasets.items():

    # Target classes
    df[target].value_counts().plot.pie(
        autopct="%.1f%%", ylabel="", title=f"{name}: Classes"
    )
    plt.savefig(figures / f"{name}_classes.png", bbox_inches="tight")
    plt.close()

    # Missing values
    missing = df.isna().mean() * 100
    missing = missing[missing > 0].sort_values()

    if not missing.empty:
        missing.plot.barh(title=f"{name}: Missing Values")
        plt.xlabel("Missing values (%)")
    else:
        plt.figure()
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
        plt.axis("off")
        plt.title(f"{name}: Missing Values")

    plt.savefig(figures / f"{name}_missing.png", bbox_inches="tight")
    plt.close()

    # Numerical features
    numeric = df.drop(columns=target).select_dtypes("number")
    numeric = numeric.where(numeric > 0)

    numeric.plot.box(logy=True, title=f"{name}: Feature Distributions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures / f"{name}_boxplot.png", bbox_inches="tight")
    plt.close()