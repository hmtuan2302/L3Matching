from __future__ import annotations

from fastapi import APIRouter
from shared.logging import get_logger

from .add import add_router

manager_router = APIRouter(prefix='/v1')
logger = get_logger(__name__)

manager_router.include_router(add_router, tags=['Files'])

@manager_router.get('/healthz')
async def healthz():
    return {'status': 'ok'}
