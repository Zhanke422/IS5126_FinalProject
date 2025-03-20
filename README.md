# IS5126_FinalProject (Twitter Sentiment Analysis)

## Project Proposal
https://docs.google.com/document/d/11AiB-DH1y1JmC6Exy9ueZtsMPgN456oMRJLmnWP_-zs/edit?tab=t.0

## Dataset can be found here
https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data

## Use the split feature files

The feature file have been split because its size exceeded limitation.
We should combine them before using.

```python
# load and combine features
part1 = np.load("word2vec_features_part1.npy")
part2 = np.load("word2vec_features_part2.npy")
features = np.vstack([part1, part2])

# load sentiment labels
labels = np.load("sentiment_labels.npy")
```
```

