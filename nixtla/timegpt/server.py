# A simple wrapper around the TimeGPT model from Nixtla to organize it for use with Ouro

import json
import os
from typing import List, Optional

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
    description="API for TimeGPT forecast integrated with Ouro",
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
        assert freq is not None, "Could not infer frequency"

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
    summary="Generate a forecast and a summary report with the TimeGPT model",
)
@ouro_field("x-ouro-input-asset-type", "dataset")
@ouro_field("x-ouro-output-asset-type", "post")
async def forecast_report(
    body: ForecastRequest, authorization: str | None = Header(default=None)
):
    try:
        api_key = authorization.split(" ")[1] if authorization else None
        client = NixtlaClient(api_key=api_key)
        if not client.validate_api_key():
            raise HTTPException(status_code=401, detail="Invalid API key")

        ouro = Ouro(
            api_key="71af2ffd480db93711239f859ac006e00f87c5e6c1d345ac0e297ca6705ba75986d9a237c7a81e1fe6a5ef85e3936db2ce86a41aa5d9bb0909b26e9c1807c4a8"
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

        data = data.reset_index()

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

        post.new_header(level=2, text="Forecast")
        post.new_code_block(
            f"SELECT * FROM datasets.{saved_forecast['metadata']['table_name']}",
            language="sql",
        )

        post.new_inline_asset(
            id=saved_forecast["id"],
            asset_type="dataset",
            view_mode="chart",
        )

        return {
            "post": {
                "name": f"Forecasted {body.dataset.name} {pd.Timestamp.now().timestamp()}",
                "description": f"Forecasting {horizon} steps into the future using the NHITS model.",
                "content": {"json": post.json, "text": post.text},
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", port=8005, reload=True)
