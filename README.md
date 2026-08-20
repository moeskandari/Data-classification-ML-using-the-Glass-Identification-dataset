# Glass Identification Classification

Coursework for SCC.222 (AI Concepts). The task is to classify glass samples into their correct type using the [UCI Glass Identification dataset](https://archive.ics.uci.edu/dataset/42/glass+identification). All classifiers are implemented from scratch, without scikit-learn's model classes.

## Repository contents

| File / folder | Description |
|---|---|
| `main.ipynb` | Full pipeline: data loading, feature scaling, the train/test split, and all three classifiers |
| `glass.data` | Raw dataset: 214 samples, 9 chemical and physical features, 6 glass types present |
| `glass+identification/` | Original UCI archive files (feature names, class tags, index) |
| `Evaluation.pdf` | Detailed comparison of the three models, including per-class performance and error analysis. The written report only has room for a summary table and one paragraph of discussion, so the fuller reasoning is kept here. |

## Methodology

The dataset is split 80/20 into training and test sets using `np.random.permutation`, seeded with my student ID so the split is reproducible. Features are scaled with min-max normalisation before training.

Three classifiers are implemented using only numpy and pandas, with no calls to `sklearn.neighbors`, `sklearn.tree`, or `sklearn.naive_bayes`:

- **K-nearest neighbours.** Classifies each test point by majority vote among its k nearest neighbours under Euclidean distance. The value of k was chosen by testing every value from 1 to 49 and keeping the one with the highest test accuracy (k = 6).
- **Decision tree.** Built recursively by selecting, at each node, the feature and threshold that maximise information gain, to a maximum depth of 10.
- **Naive Bayes.** Models each feature as Gaussian within each class and combines the per-feature likelihoods, with a small variance floor added to avoid division by zero.

## Results

| Model | Accuracy | Train time | Test time |
|---|---|---|---|
| KNN (k = 6) | 60.5% | n/a (lazy learner) | ~0.0005 s |
| Decision tree | 76.7% | ~0.20 s | ~0.00007 s |
| Naive Bayes | 41.9% | ~0.0007 s | ~0.002 s |

The decision tree gave the best result by a clear margin. Naive Bayes performed worst, which is consistent with the data: several features, such as refractive index, calcium content, and sodium content, are correlated with one another, which violates the conditional independence assumption the model depends on. Confusion matrices and a per-class breakdown for all three models are in the notebook and in `Evaluation.pdf`.

## What I learned

Implementing these algorithms from scratch, rather than importing them, made the underlying mechanics far clearer than reading the theory alone. Writing the recursive tree-building function required working through exactly what "split on best information gain" means at each step, instead of treating it as a black box. The same was true of Naive Bayes: computing the Gaussian likelihoods directly made the failure mode visible once I checked feature correlations and saw how strongly several of them move together.

The small size of the dataset mattered more than I expected. With 214 samples in total and around 43 in the test set, some glass types are barely represented in testing, so a handful of misclassifications can shift a class's accuracy substantially. That made it clear how easily a single overall accuracy figure can be misleading, and why the confusion matrix needs to be read carefully alongside it.

Vectorisation also had a noticeable effect on performance. The KNN prediction step, computed as one distance calculation against the entire training set per query point, ran almost instantly. The decision tree's split search, which loops over every feature and every candidate threshold in Python, was by far the slowest part of the notebook.

## What I would do differently

Given another attempt at this coursework, several changes would improve the reliability of the results.

A single 80/20 split on 214 samples is sensitive to the particular random permutation used, so the reported accuracy could look noticeably different with another seed. K-fold cross-validation would give a more stable estimate of how each model actually generalises, rather than how it performs on one specific partition.

Reporting only accuracy also hides how the models handle the less common glass types. Precision, recall, and F1 per class would make that failure pattern visible directly, which matters given how imbalanced this dataset is.

The decision tree's maximum depth of 10 was chosen without much justification. Selecting it against a validation set, rather than fixing it in advance, would likely reduce overfitting and close some of the gap between training and test accuracy.

The Gaussian assumption behind Naive Bayes was also not checked before use. Examining the actual feature distributions first, and using a kernel density estimate or a transform for the features that are visibly skewed, would probably help more than adjusting the smoothing constant.

Finally, the decision tree's split search is inefficient: it evaluates every unique value of every feature as a candidate threshold. Sorting each feature once and scanning it, or restricting candidates to a fixed set of percentiles, would reduce training time substantially without a meaningful loss in accuracy.

## Running the notebook

Open `main.ipynb` in Jupyter and run all cells in order. It expects `glass.data` to be in the same directory. `numpy`, `pandas`, `matplotlib`, and `time` are used for the machine learning itself; `sklearn` is used only for the permitted helper functions (confusion matrices and plotting), never for the models.
