# A simple wrapper around the TimeGPT model from Nixtla to organize it for use with Ouro
import json
import math
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import uvicorn
from autots import infer_frequency
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from nixtla import NixtlaClient
from ouro.utils import get_custom_openapi, ouro_field
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_percentage_error

from ouro import Ouro

load_dotenv()  # take environment variables from .env.


class DatasetMetadata(BaseModel):
    table_name: str


class Dataset(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = ""
    metadata: DatasetMetadata
    data: List[dict]


class ForecastConfig(BaseModel):
    horizon: int
    date_column: str
    value_column: str


class ForecastRequest(BaseModel):
    dataset: Dataset
    config: ForecastConfig


# Initialize FastAPI app
app = FastAPI(
    title="Nixtla Forecast API",
    description="Forecasting with TimeGPT integrated with Ouro",
    servers=[
        {
            "url": "https://nixtla.ouro.foundation",
            "description": "Production environment",
        },
    ],
    terms_of_service="",
    contact={
        "name": "Chronos",
        "url": "https://ouro.foundation/app/users/chronos",
        "email": "chronos@ouro.foundation",
    },
)

# Set the custom openapi function
app.openapi = get_custom_openapi(app, get_openapi)


# Allow origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.environ.get("OURO_FRONTEND_URL", "http://localhost:3000"),
        os.environ.get("OURO_BACKEND_URL", "http://localhost:8003"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/forecast", summary="Generate a forecast using the TimeGPT model")
@ouro_field("x-ouro-input-asset-type", "dataset")
@ouro_field("x-ouro-output-asset-type", "dataset")
async def forecast(
    body: ForecastRequest, authorization: str | None = Header(default=None)
):
    try:
        api_key = authorization.split(" ")[1] if authorization else None
        client = NixtlaClient(api_key=api_key)
        if not client.validate_api_key():
            raise HTTPException(status_code=401, detail="Invalid API key")

        data = pd.DataFrame(body.dataset.data)

        horizon = body.config.horizon
        date_column = body.config.date_column
        value_column = body.config.value_column

        assert date_column in data.columns, f'Date column "{date_column}" not found'
        assert value_column in data.columns, f'Value column "{value_column}" not found'

        data[date_column] = pd.to_datetime(data[date_column])

        data.rename(columns={date_column: "ds", value_column: "y"}, inplace=True)
        data["unique_id"] = 1
        data = data[["unique_id", "ds", "y"]]
        data = data.set_index("ds")

        freq = infer_frequency(data)
        assert freq is not None, "Could not data infer frequency"

        # Resample data to the inferred frequency
        data = data.resample(freq).mean().reset_index()
        # FIll in missing values
        data = data.interpolate(method="linear")

        data = data.reset_index()

        Y_hat_df = client.forecast(data, h=horizon, model="timegpt-1-long-horizon")
        Y_hat_df = Y_hat_df[["ds", "TimeGPT"]]
        forecast = Y_hat_df.to_dict(orient="records")

        return {
            "dataset": {
                "name": f"Forecasted {body.dataset.name} {pd.Timestamp.now().timestamp()}",
                "description": f"Forecasting {horizon} steps into the future using the TimeGPT model.",
                "data": forecast,
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/forecast/report",
    summary="Generate a forecast report with the TimeGPT model",
)
@ouro_field("x-ouro-input-asset-type", "dataset")
@ouro_field("x-ouro-output-asset-type", "post")
async def forecast_report(
    body: ForecastRequest, authorization: str | None = Header(default=None)
):
    try:
        api_key = authorization.split(" ")[1] if authorization else None
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing Nixtla API key")
        client = NixtlaClient(api_key=api_key)
        if not client.validate_api_key():
            raise HTTPException(status_code=401, detail="Invalid Nixtla API key")

        ouro = Ouro(
            api_key=os.getenv("OURO_API_KEY"),
        )

        data = pd.DataFrame(body.dataset.data)

        horizon = body.config.horizon
        date_column = body.config.date_column
        value_column = body.config.value_column

        assert date_column in data.columns, f'Date column "{date_column}" not found'
        assert value_column in data.columns, f'Value column "{value_column}" not found'

        data[date_column] = pd.to_datetime(data[date_column])

        data.rename(columns={date_column: "ds", value_column: "y"}, inplace=True)
        data["unique_id"] = 1
        data = data[["unique_id", "ds", "y"]]
        data = data.set_index("ds")

        freq = infer_frequency(data)
        assert freq is not None, "Could not infer frequency"

        # Resample data to the inferred frequency
        data = data.resample(freq).mean().reset_index()
        # FIll in missing values
        data = data.interpolate(method="linear")

        # Cross-validation
        # Calculate how many windows we can do
        n_windows = np.min([(len(data) - horizon * 3) // horizon, 4])
        cv_df = client.cross_validation(
            data,
            h=horizon,
            n_windows=n_windows,
            time_col="ds",
            target_col="y",
            freq=freq,
        )

        errors = []
        cutoffs = cv_df["cutoff"].unique()
        for cutoff in cutoffs:
            fold_df = cv_df.query("cutoff == @cutoff")

            # Calculate the error metrics
            mape = mean_absolute_percentage_error(fold_df["y"], fold_df["TimeGPT"])
            errors.append(mape)
        average_mape = np.mean(errors)

        Y_hat_df = client.forecast(data, h=horizon)
        Y_hat_df = Y_hat_df[["ds", "TimeGPT"]]
        Y_hat_df = Y_hat_df.rename(columns={"TimeGPT": "y"})
        Y_hat_df["ds"] = pd.to_datetime(Y_hat_df["ds"])
        # forecast = Y_hat_df.to_dict(orient="records")

        Y_hat_df["type"] = "forecast"
        data["type"] = "train"
        full_df = pd.concat([data[["ds", "y", "type"]], Y_hat_df])

        saved_forecast = ouro.elements.earth.datasets.create(
            name=f"Forecasted {body.dataset.name} {pd.Timestamp.now().timestamp()}",
            description=f"Forecasting {horizon} steps into the future using the TimeGPT model.",
            visibility="private",
            data=full_df,
        )

        post = ouro.elements.air.Editor()
        # Add a title to the report
        post.new_header(level=1, text=f"{body.dataset.name} Forecast")
        post.new_inline_asset(
            id=saved_forecast["id"],
            asset_type="dataset",
            view_mode="chart",
        )
        post.new_code_block(
            f"SELECT * FROM datasets.{saved_forecast['metadata']['table_name']}",
            language="sql",
        )
        # Error metrics
        post.new_header(level=2, text="Error metrics")
        post.new_paragraph(
            f"In order to evaluate the predictive power and confidence in the forecast, the model was cross-validated {n_windows} times. "
            f"Average mean absolute percentage error (MAPE) across all folds was calculated to be {(average_mape * 100).round(2)}%. "
            + "The lower the MAPE, the better the model is at predicting the future."
        )
        post.new_header(level=2, text="Report details")
        post.new_paragraph(
            "This report was generated by Nixtla's TimeGPT forecasting route. "
            "The forecast was run with the following config:"
        )
        post.new_code_block(
            json.dumps(body.config.model_dump(), indent=2),
            language="json",
        )
        post.new_header(level=2, text="Training data")
        post.new_paragraph(
            "Training data comes from the Dataset you passed in to the config. "
            "You can query for the data used with the following SQL query:"
        )
        post.new_code_block(
            f"SELECT * FROM datasets.{body.dataset.metadata.table_name}",
            language="sql",
        )
        post.new_table(data.head())

        return {
            "post": {
                "name": f"Forecasted {body.dataset.name} {pd.Timestamp.now().timestamp()}",
                "description": f"Forecasting {horizon} steps into the future using the TimeGPT model from @nixtla",
                "content": {"json": post.json, "text": post.text},
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", port=8005, reload=True)
