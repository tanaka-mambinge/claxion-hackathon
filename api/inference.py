import pickle
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
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

from api.sample_data import sample_data
from api.utils import clipper, standardize_columns

router = APIRouter(
    prefix="/api/v1/model/inference",
)


def load_data(file: UploadFile) -> pl.DataFrame:
    file_bytes = file.file.read()
    buffer = StringIO(file_bytes.decode("utf-8"))
    df = pl.read_csv(buffer)
    return df


def preprocess_data(df: pl.DataFrame) -> tuple:
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

    # normalize data
    std_cols = [
        "number_of_defaults",
        "age",
        "salary",
    ]
    min_max_cols = [col for col in X.columns if col not in std_cols]
    transformer = ColumnTransformer(
        [
            ("minmax", MinMaxScaler(), min_max_cols),
            ("zscore", StandardScaler(), std_cols),
        ],
    )

    transformer = transformer.fit(X)
    X = transformer.transform(X)

    return X


@router.post("", status_code=status.HTTP_200_OK)
async def model_inference(file: UploadFile = File(...)):
    df = load_data(file)
    X = preprocess_data(df)
    model = pickle.load(open("artifacts/random-forest.sav", "rb"))
    y_pred: np.ndarray = model.predict(X)
    return {"results": y_pred.tolist()}
