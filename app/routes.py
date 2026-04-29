from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from .constants import DEFAULT_USER_ID
from .bootstrap import templates
from .data_store import load_app_data
from .services import home_context, recommendations_context

# Create the FastAPI router that connects URL paths to page functions.
router = APIRouter()
# Load processed recommendation data once when the web app starts.
data = load_app_data()

# Home page route.
# It displays the project summary, dataset statistics, and sample recommendations.
@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    context = {"request": request, **home_context(data)}
    return templates.TemplateResponse("index.html", context)

# Recommendations page route.
# It shows the final recommended products for the selected user ID.
@router.get("/recommendations", response_class=HTMLResponse)
def recommendation_page(request: Request, user_id: int = Query(DEFAULT_USER_ID)) -> HTMLResponse:
    context = {"request": request, **recommendations_context(data, user_id)}
    return templates.TemplateResponse("recommendations.html", context)
