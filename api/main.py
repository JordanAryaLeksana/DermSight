from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import health, skin, recommendation


app = FastAPI(
    title="DermSight API",
    description="Skin disease classification and LLM recommendation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dermsight-v1.streamlit.app"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("CORS middleware configured to allow all origins, methods, and headers.")
@app.get("/")
def root():
    return {"message": "Welcome to the DermSight API! Visit /docs for API documentation."}

print("Root endpoint configured at /")
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)


print("Health router included at /health")
app.include_router(
    skin.router,
    prefix="/skin",
    tags=["Skin Disease"],
)

print("Skin router included at /skin")
app.include_router(
    recommendation.router,
    prefix="/recommendation",
    tags=["Recommendation"],
)
print("Recommendation router included at /recommendation")
