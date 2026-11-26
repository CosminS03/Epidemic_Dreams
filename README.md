# Epidemic_Dreams

**Goal:** reproduce two figures from the paper (`paper.pdf`):
- **Figure 2:** scatter of condition rank in tweets (x) vs. dream reports (y), colored by class.
- **Figure 3:** core of the dream condition co-occurrence network, sized by PageRank, colored by class.

## Setup

Requires Python 3.13. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy matplotlib networkx jupyter
```

All input data is in `data/` (field descriptions in `data/README.txt`). Run every command below from the repo root, since the code uses relative paths.

## Figure 2

```bash
python figure2.py
```

Writes `results/Figure2.png` and prints how many conditions fall in each class.

How it works: the conditions from `equally_prevalent.csv`, `more_prev_in_dreams_covid.csv` and `more_prev_in_twitter_covid.csv` start out as "equal". A regression line of dream rank on Twitter rank is fitted on the equal ones. Conditions with a residual of 1.5 standard deviations or more are relabelled waking (red) or dreams (blue). This repeats until nothing changes or R² reaches 0.98.

Expected output: 255 waking, 95 equal, 57 dreams. The paper reports 277 / 63 / 57, so the waking/equal split differs slightly.

## Figure 3

Open `figure3.ipynb` (`jupyter notebook figure3.ipynb`, or in VS Code with the `.venv` kernel) and run both cells in order.

1. **Cell 1** builds the network and draws it:
   - Keeps only pairs that co-occur significantly more often than chance (phi coefficient, one-sided p < 0.01, N = 2888 dream reports). This leaves 3,787 edges; the paper reports 3,759.
   - Removes COVID itself: any condition containing `covid`, `corona`, `pandem` or `virus`.
   - Colors by class, as in the paper's caption: orange = equally prevalent, green = typical of waking discussions, blue = typical of dream reports. "Only in dreams" counts as dreams and "only in waking" as waking.
   - Keeps the core: nodes with 8+ edges (`MIN_DEGREE`), then drops the bottom 10% by PageRank.
   - Sizes nodes by PageRank (scaled to `MIN_SIZE`–`MAX_SIZE`) and lays them out with a spring layout.

   Expected printout: `Graph with 167 nodes and 1037 edges {'dreams': 112, 'waking': 30, 'equal': 25}`.

2. **Cell 2** writes `figure3_core.gexf` for Gephi, with each node's color, size, class (`cls`) and `pagerank`, and edge weights.

The notebook only shows the plot; it doesn't save an image. `results/Figure3.png` was exported separately.

### Finishing the layout in Gephi

The published figure was laid out by hand in Gephi, so it can't be reproduced exactly. To get close:

1. In Gephi, **File → Open** `figure3_core.gexf`. Colors and sizes load from the file, so don't re-apply Partition or Ranking in Appearance.
2. In **Layout**, run **Community Clusters**, then **Label Adjust** to stop labels overlapping.
3. Show labels with the **T** button, adjust nodes if needed, and export with **File → Export → SVG/PDF/PNG**.

## Differences from the paper

- **Figure 2:** the class counts differ slightly, as noted above.
- **Figure 3:**
  - The paper also requires the lower bound of the relative risk to be above 1. With these counts that removes about 480 more edges than the paper reports, so it isn't applied.
  - The paper uses a core of 5+ edges, which here leaves 303 nodes after the PageRank cut. 8+ gives a core closer in size to the published figure.
  - The paper's exact layout and group labels came from manual editing, so they can't be reproduced.
