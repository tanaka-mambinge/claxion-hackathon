import pickle
from io import StringIO

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from api.utils import clipper, standardize_columns

router = APIRouter(
    prefix="/api/v1/model/train",
)


def load_data(file: UploadFile) -> pl.DataFrame:
    file_bytes = file.file.read()
    buffer = StringIO(file_bytes.decode("utf-8"))
    df = pl.read_csv(buffer)
    return df


def get_mutual_info(df: pl.DataFrame) -> pl.DataFrame:
    enc = OrdinalEncoder()
    df = df.with_columns(
        pl.Series("gender", enc.fit_transform(df.select("gender")).flatten()),
        pl.Series("job", enc.fit_transform(df.select("gender")).flatten()),
        pl.Series("location", enc.fit_transform(df.select("gender")).flatten()),
        pl.Series("marital_status", enc.fit_transform(df.select("gender")).flatten()),
    )

    mi = mutual_info_classif(
        df.drop("defaulted"),
        df.select("defaulted").to_series(),
        random_state=42,
        n_jobs=-1,
    )

    mi = sorted(zip(mi, df.drop("defaulted").columns))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax = sns.barplot(
        x=[x[0] for x in mi],
        y=[x[1] for x in mi],
        ax=ax,
    )
    ax.set(xlabel="Mutual Information", ylabel="Features")


def preprocess_data(df: pl.DataFrame, n_train: float, n_test: float) -> tuple:
    df = df.rename(standardize_columns)

    # one hot encode data
    df = df.with_columns(df.select("location").to_dummies())
    df = df.with_columns(df.select("job").to_dummies())
    df = df.with_columns(df.select("marital_status").to_dummies())
    df = df.with_columns(df.select("gender").to_dummies())

    # clip numerical values
    df.with_columns(
        df.select(
            pl.col("outstanding_balance").map_elements(
                lambda x: clipper(x, 15000, 80000), return_dtype=pl.datatypes.Float32
            )
        )
    )

    df = df.with_columns(
        df.select(
            pl.col("loan_amount").map_elements(
                lambda x: clipper(x, 0, 70000), return_dtype=pl.datatypes.Float32
            )
        )
    )

    # train test split
    X = df.select(
        pl.col(
            [
                "is_employed",
                "loan_amount",
                "number_of_defaults",
                "outstanding_balance",
                "interest_rate",
                "age",
                "salary",
                "gender",
                "location",
                "job",
                "marital_status",
            ]
        )
    )
    y = df.select(pl.col("defaulted"))

    split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_train,
        test_size=n_test,
        random_state=42,
    )

    train_idx, test_idx = next(split.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # normalize data
    std_cols = [
        "number_of_defaults",
        "age",
        "salary",
    ]
    min_max_cols = [col for col in X_train.columns if col not in std_cols]
    transformer = ColumnTransformer(
        [
            ("minmax", MinMaxScaler(), min_max_cols),
            ("zscore", StandardScaler(), std_cols),
        ],
    )

    transformer = transformer.fit(X_train)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
    y_train = y_train.to_numpy().flatten()
    y_test = y_test.to_numpy().flatten()

    return X_train, X_test, y_train, y_test, X.columns


def train_and_eval_model(
    X_train, X_test, y_train, y_test, columns: list[str], file_name: str
) -> dict:
    params = {
        "n_estimators": [160, 180, 200, 220],
        "class_weight": [{0: 0.2, 1: 0.8}, {0: 0.3, 1: 0.7}],
    }

    rf = RandomForestClassifier(random_state=42)
    rf_search = GridSearchCV(
        rf, params, n_jobs=-1, cv=4, verbose=2, scoring="f1_weighted"
    )
    rf_search.fit(X_train, y_train)
    print(f"Best hyperparams: {rf_search.best_params_}")
    print(f"Best score: {rf_search.best_score_}")

    rf = RandomForestClassifier(**rf_search.best_params_, n_jobs=-1)
    rf.fit(X_train, y_train)

    pickle.dump(rf, open(f"artifacts/{file_name}.sav", "wb"))

    # Save image
    rf_fi = sorted(zip(rf.feature_importances_, columns))
    fig, ax = plt.subplots(figsize=(6, 12))
    ax = sns.barplot(
        x=[x[0] for x in rf_fi],
        y=[x[1] for x in rf_fi],
        ax=ax,
    )
    ax.set(
        xlabel="Feature Importance",
        ylabel="Features",
        title="Feature importance (Random Forest)",
    )

    y_pred = rf.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)


@router.post("", status_code=status.HTTP_201_CREATED)
async def train_model(file: UploadFile = File(...)):
    n_train: float = 0.8
    n_test: float = 0.2
    df = load_data(file)
    get_mutual_info(df)
    X_train, X_test, y_train, y_test, columns = preprocess_data(df, n_train, n_test)
    report = train_and_eval_model(
        X_train, X_test, y_train, y_test, columns, "random-forest-01"
    )
    return report
