from __future__ import annotations

from api.helpers import LoggingMiddleware
from api.routers import manager_router
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from shared.logging import get_logger
from shared.logging import setup_logging

setup_logging(json_logs=True)
logger = get_logger('api')

app = FastAPI(title='Test', version='2.0.0')
origins = [
    'http://localhost',
    'http://localhost:8080',
]


# add middleware to generate correlation id
app.add_middleware(LoggingMiddleware, logger=logger)
app.add_middleware(CorrelationIdMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(
    manager_router,
)

instrumentator = Instrumentator(
    should_ignore_untemplated=True,
    excluded_handlers=['/metrics', '/docs', '/healthz'],
)
instrumentator.instrument(app).expose(app)
