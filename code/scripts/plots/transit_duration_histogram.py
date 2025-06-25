import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])

"""Plot the transit duration histogram. Calculate percentile.
"""

p = 0.95  # Percentile
disposition = "CP"  # Disposition of the exoplanet

# Read the project TESS candidates table
df = pd.read_csv(
    "../../star_names/nasa_exo_archive/project_candidates_TESS.csv"
)
df = df.drop(columns=["pl_trandurherr1", "pl_trandurherr2", "pl_trandurhlim"])
df = df.rename(
    columns={"tfopwg_disp": "disposition", "pl_trandurh": "transit_duration_h"}
)

# Filter by disposition
single_disposition_only = df[df["disposition"] == "CP"].reset_index(drop=True)

# Plot the histogram
fig, ax = plt.subplots(dpi=300)
pd.DataFrame.hist(
    single_disposition_only,
    "transit_duration_h",
    bins=list(range(0, int(max(single_disposition_only["transit_duration_h"])) + 1)),
    ax=ax,
    color="blue",
    rwidth=0.9,
)

ax.set_title("Histogram with transit durations")
ax.set_xlabel("Transit Duration [h]")
ax.set_ylabel("Number of exoplanets per bin")

plt.tight_layout()
plt.savefig("./figures/histogram_transit_duration.pdf")
plt.show()

# Compute the percentile
durations_only = single_disposition_only["transit_duration_h"]
durations_only.quantile(q=0.99)
