# This script trains the fraud detection model and saves it so the API can use it.
# Run it with:  python train_model.py
# It reads insurance_claims.csv, trains a Random Forest and saves model.pkl

import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# the columns I decided to use as inputs (I didn't use all 40, a lot of them
# like policy_number or zip code didn't really make sense as features)
NUMERIC = [
    "months_as_customer",
    "age",
    "policy_annual_premium",
    "policy_deductable",
    "umbrella_limit",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "total_claim_amount",
    "incident_hour_of_the_day",
]

CATEGORICAL = [
    "insured_sex",
    "insured_relationship",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "property_damage",
    "police_report_available",
]

TARGET = "fraud_reported"


def main():
    df = pd.read_csv("insurance_claims.csv")
    print("loaded data:", df.shape)

    # target is Y / N in the csv, change it to 1 / 0
    y = df[TARGET].map({"Y": 1, "N": 0})
    X = df[NUMERIC + CATEGORICAL]

    # some categorical columns have "?" instead of a real value.
    # I just leave it as its own category, the one hot encoder handles it.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # one hot encode the text columns, leave the numbers as they are
    # (random forest doesn't need scaling so I didn't scale)
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="passthrough",
    )

    # class_weight balanced because there are way more normal claims than fraud ones
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
    )

    clf = Pipeline(steps=[("pre", pre), ("model", model)])
    clf.fit(X_train, y_train)

    # check how it did
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    print("\naccuracy:", round(accuracy_score(y_test, preds), 3))
    print("roc auc :", round(roc_auc_score(y_test, probs), 3))
    print("\n", classification_report(y_test, preds))

    # save the trained pipeline so the api can load it
    joblib.dump(clf, "model.pkl")

    # also save the feature info, the frontend uses this to build the dropdowns
    feature_info = {
        "numeric": NUMERIC,
        "categorical": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL},
    }
    with open("feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    print("\nsaved model.pkl and feature_info.json")


if __name__ == "__main__":
    main()
