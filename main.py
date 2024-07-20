from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import inference, model_train

app = FastAPI()
app.include_router(model_train.router)
app.include_router(inference.router)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
