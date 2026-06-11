# Feature Selection Report — Agent 3

This stage does not permanently remove columns yet. Instead, we rank features by **permutation importance**
on a holdout split (accuracy metric) to guide Agent 4 modeling.

Holdout accuracy (quick check): **0.8108**
Model: HistGradientBoostingClassifier + OneHotEncoder
