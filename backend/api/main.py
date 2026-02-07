import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  

# 1. Import your routers
import sys
from pathlib import Path

# Add the backend directory to sys.path so we can import api module
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from api.routers import assets
from api.routers import social_sentiment

# 2. Create the app instance
app = FastAPI(title="Quant Factor API", version="1.0.0")

# 3. Add CORS Middleware (Add this now so you don't forget)
origins = [
    "http://localhost:5173",  # Your local Vite/React app
    "https://alphaone.run.place" # Your future domain
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. "Mount" your routers under the /api/v1 prefix
app.include_router(assets.router, prefix="/api/v1")
app.include_router(
    social_sentiment.router, 
    prefix="/api/v1/signals"
)

# 5. Define your single root endpoint (for health checks)
@app.get("/")
async def read_root():
    return {"status": "API is running"}

# 6. Run the app (This must be at the END)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)