from .health import router as health_router
from .skin import router as skin_router
from .recommendation import router as recommendation_router
__all__ = [
    "health_router",
    "skin_router",
    "recommendation_router",
]