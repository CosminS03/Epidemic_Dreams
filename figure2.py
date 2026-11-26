import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

df = pd.concat(
    [
        pd.read_csv("data/equally_prevalent.csv"),
        pd.read_csv("data/more_prev_in_dreams_covid.csv"),
        pd.read_csv("data/more_prev_in_twitter_covid.csv"),
    ],
    ignore_index=True,
)[["rank_twitter_covid", "rank_dreams_covid"]]

df["Class"] = "e"

size = 0
r2 = 0
res = pd.DataFrame(
    columns=["rank_twitter_covd", "rank_dreams_covid", "Class", "residuals"]
)
while size != (df["Class"] == "e").sum() and r2 < 0.98:
    df = df[df["Class"] == "e"]
    size = len(df)
    result = stats.linregress(df["rank_twitter_covid"], df["rank_dreams_covid"])
    df["residuals"] = df["rank_dreams_covid"] - (
        result.slope * df["rank_twitter_covid"] + result.intercept
    )

    mu, sigma = df["residuals"].mean(), df["residuals"].std()
    x = df["residuals"] - mu

    df.loc[x >= 1.5 * sigma, "Class"] = "w"
    df.loc[x <= -1.5 * sigma, "Class"] = "d"

    rows = df[(df["Class"] == "w") | (df["Class"] == "d")]
    res = pd.concat([res, rows], ignore_index=True)
    r2 = result.rvalue**2

df = pd.concat([df, res], ignore_index=True)

colors = {"e": "#FFF000", "w": "#FF0000", "d": "#0000FF"}
fig, ax = plt.subplots()
ax.scatter(df["rank_twitter_covid"], df["rank_dreams_covid"], c=df["Class"].map(colors))

x_line = np.array([df["rank_twitter_covid"].min(), df["rank_twitter_covid"].max()])
ax.plot(x_line, result.slope * x_line + result.intercept, color="black")
fig.savefig("results/Figure2.png")

print(df["Class"].value_counts())
