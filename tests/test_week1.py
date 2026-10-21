"""Tests for Week 1 functional requirements FR-1.1 to FR-1.7.

Run from the repo root:  python tests/test_week1.py   (or: pytest tests/)
"""
import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data_pipeline as dp  # noqa: E402
import risk_model as rm  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "patient_records.csv")


def test_fr_1_1_missing_column_raises_named_error():
    df = pd.read_csv(DATA, keep_default_na=False).drop(columns=["Age"])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.csv")
        df.to_csv(p, index=False)
        try:
            dp.load_raw(p)
        except dp.MissingColumnsError as e:
            assert "Age" in str(e)
        else:
            raise AssertionError("MissingColumnsError was not raised")


def test_fr_1_2_cleaning_leaves_no_missing_values_and_keeps_none_category():
    raw = dp.load_raw(DATA)
    assert raw.isnull().sum().sum() > 0
    n_none = (raw[dp.CATEGORICAL_COL] == "None").sum()
    assert n_none > 250, "'None' must stay a real category, not become NaN"
    clean = dp.clean(raw)
    assert (clean.isnull().sum() == 0).all()
    assert (clean[dp.CATEGORICAL_COL] == "None").sum() >= n_none


def test_fr_1_3_scaling_gives_zero_mean_unit_std_on_train():
    df = dp.encode_categorical(dp.clean(dp.load_raw(DATA)))
    X_train, _, _, _ = dp.split_data(df)
    Xs = dp.apply_scaler(X_train, dp.fit_scaler(X_train))
    for col in dp.NUMERIC_COLS:
        assert abs(Xs[col].mean()) < 1e-6
        assert abs(Xs[col].std(ddof=0) - 1) < 1e-6


def test_fr_1_4_split_is_80_20_stratified_and_reproducible():
    df = dp.encode_categorical(dp.clean(dp.load_raw(DATA)))
    a = dp.split_data(df, seed=42)
    b = dp.split_data(df, seed=42)
    assert len(a[0]) == 400 and len(a[1]) == 100
    assert a[0].index.equals(b[0].index) and a[1].index.equals(b[1].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.03  # class balance preserved


def test_fr_1_5_to_1_7_training_reports_and_serialization():
    with tempfile.TemporaryDirectory() as d:
        m = rm.run_pipeline(DATA, out_dir=d, clean_csv_path=None, verbose=False)
        # FR-1.5: both models trained
        assert set(m["models"]) == {"logistic_regression", "random_forest"}
        # FR-1.6: reports + confusion matrices saved for both models
        for name in m["models"]:
            assert os.path.getsize(os.path.join(d, "figures", f"classification_report_{name}.txt")) > 0
            assert os.path.getsize(os.path.join(d, "figures", f"confusion_matrix_{name}.png")) > 0
        # FR-1.7: best model by F1 is saved and reloads with identical predictions
        best = max(m["models"], key=lambda n: m["models"][n]["f1_high_risk"])
        assert m["selected_model"] == best
        model = joblib.load(os.path.join(d, "risk_model.joblib"))
        pre = joblib.load(os.path.join(d, "preprocessor.joblib"))
        df = dp.encode_categorical(dp.clean(dp.load_raw(DATA)))
        _, X_test, _, _ = dp.split_data(df)
        preds = model.predict(dp.apply_scaler(X_test, pre["scaler"]))
        saved = pd.read_csv(os.path.join(d, "test_predictions.csv"))["y_pred"].values
        assert np.array_equal(preds, saved)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} Week 1 tests passed.")
