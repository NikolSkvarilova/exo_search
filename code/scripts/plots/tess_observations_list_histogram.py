from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])

"""Load the SPOC star lists, create a selection and a histogram.
"""

# Directory with SPOC star names
spoc_dir = Path("../../star_names/tess_spoc/")
filenames = list(spoc_dir.iterdir())

# Load the dataframes
dfs = []
for fn in filenames:
    df = pd.read_csv(fn)
    dfs.append(df)

# Merge dataframes
df_merged = pd.concat(dfs)
# Only take the name
df_merged = df_merged[["#TIC_ID"]]

# Count how many observations there are per star
df_merged["count"] = 1
df_merged = df_merged.reset_index()

df_grouped = df_merged.groupby(by=["#TIC_ID"]).sum().reset_index()
df_grouped = df_grouped.sort_values("count", ascending=False)

# Plot the histogram
plt.figure(dpi=300)
plt.title("Observed sectors per star (TESS - SPOC)")
plt.hist(
    df_grouped["count"], bins=list(range(1, max(df_grouped["count"]))), color="blue"
)
plt.xlabel("Number of sectors")
plt.ylabel("Number of stars per bin")
plt.yscale("log")
plt.tight_layout()
plt.savefig("figures/histogram_sectors.pdf")
plt.show()
plt.close()

# Save the data
# Rename the TICs
# df_grouped["star_name"] = df_grouped["#TIC_ID"].apply(lambda x: f"TIC {x}")
# df_grouped = df_grouped[["star_name", "count"]]

# Save all
# df_grouped.to_csv("../star_names/tess/spoc_by_sectors.csv", index=False)

# Saving a selection of the stars:

# selection = df_grouped.iloc[0:1000]
# selection.to_csv("../star_names/tess/spoc_by_sectors_analysis_0-1000.csv", index=False)
