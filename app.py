from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Wine Quality Classification API")

# Setup templates (we will create a templates folder)
templates = Jinja2Templates(directory="templates")

# Try loading the model globally
try:
    model = joblib.load("model.joblib")
except Exception as e:
    model = None
    print(f"Watch out: model.joblib not found. {e}")

# Target class names for Wine dataset
wine_classes = ['Class 0 (Cultivar 1)', 'Class 1 (Cultivar 2)', 'Class 2 (Cultivar 3)']

# Define Pydantic schema for REST API requests
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the Home HTML page."""
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict_ui", response_class=HTMLResponse)
async def predict_ui(
    request: Request,
    alcohol: float = Form(...),
    malic_acid: float = Form(...),
    ash: float = Form(...),
    alcalinity_of_ash: float = Form(...),
    magnesium: float = Form(...),
    total_phenols: float = Form(...),
    flavanoids: float = Form(...),
    nonflavanoid_phenols: float = Form(...),
    proanthocyanins: float = Form(...),
    color_intensity: float = Form(...),
    hue: float = Form(...),
    od280_od315_of_diluted_wines: float = Form(...),
    proline: float = Form(...)
):
    """Handle UI form submission for prediction."""
    if model is None:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "prediction": "Model not loaded. Please train the model first.",
            "error": True
        })
        
    # Prepare features array
    features = np.array([[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
                          total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
                          color_intensity, hue, od280_od315_of_diluted_wines, proline]])
    
    # Make prediction
    prediction_idx = model.predict(features)[0]
    predicted_class = wine_classes[prediction_idx]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": f"The predicted wine class is: {predicted_class}"
    })

@app.post("/api/predict")
def predict_api(features: WineFeatures):
    """REST API endpoint for prediction."""
    if model is None:
        return {"error": "Model not loaded."}
        
    data = np.array([[
        features.alcohol, features.malic_acid, features.ash, 
        features.alcalinity_of_ash, features.magnesium, features.total_phenols, 
        features.flavanoids, features.nonflavanoid_phenols, features.proanthocyanins, 
        features.color_intensity, features.hue, features.od280_od315_of_diluted_wines, 
        features.proline
    ]])
    prediction = int(model.predict(data)[0])
    
    return {"predicted_class_index": prediction, "predicted_class_name": wine_classes[prediction]}
