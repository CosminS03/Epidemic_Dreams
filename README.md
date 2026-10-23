# Epidemic_Dreams
 
**Goal:** reproduce
- **Figure 2:** scatter of condition rank in tweets (x) vs. dream reports (y), colored by class.
- **Figure 3:** core of the dream condition co-occurrence network (nodes with 5+ edges), sized by PageRank, colored by class.

## Checklist
 
- [ ] Set up repo and Colab notebook; load all CSVs
- [ ] Check row counts: 63 equally prevalent, 277 waking, 57 dreams
- [ ] **Fig. 2:** plot `rank_twitter_covid` vs. `rank_dreams_covid`, colored by class, with regression line
- [ ] **Fig. 3:** build graph from `co_occurrence_graph.csv`, attach class labels
- [ ] Compute PageRank; keep nodes with degree ≥ 5
- [ ] Draw network (size = PageRank, color = class, edge width = count)
- [ ] Compare both figures with the originals and note differences
