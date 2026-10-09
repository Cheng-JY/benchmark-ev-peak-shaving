"""Export trained classifier and regression coefficients from a saved model bundle."""

from pathlib import Path
import joblib
import pandas as pd

bundle = joblib.load("data/training/model.joblib")
output = Path("data/training")
output.mkdir(parents=True, exist_ok=True)

for name in ["regression", "classifier"]:
    pipeline = bundle[name]
    transformer = pipeline.steps[0][1]
    estimator = pipeline.steps[-1][1]

    feature_names = transformer.get_feature_names_out()
    coefficients = estimator.coef_.reshape(-1)

    table = pd.DataFrame({
        "term": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False)

    table.to_csv(
        output / f"{name}_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n{name}: intercept = {estimator.intercept_}")
    print(table.head(20).to_string(index=False))