from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import boto3, json, os
from dotenv import load_dotenv

# Load .env
load_dotenv()
print("DEBUG ENDPOINT:", os.getenv("ENDPOINT_NAME"))

ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ✅ Correct label mapping (Linear Learner swapped 0/1)
CLASS_LABELS = {
    0: "SETOSA",
    1: "Versicolor",
    2: "Virginica"
}

def invoke_endpoint(features):
    # Convert list of floats → CSV string
    payload = ",".join(map(str, features))
    
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",   # Linear Learner expects CSV
        Body=payload
    )
    result = response["Body"].read().decode("utf-8").strip()

    try:
        data = json.loads(result)
        if "predictions" in data:
            pred = data["predictions"][0]

            # 🔍 Debug: print raw result
            print("DEBUG RAW PREDICTION:", pred)

            if "predicted_label" in pred:
                label = int(pred["predicted_label"])
                return f"Predicted Class: {CLASS_LABELS.get(label, label)} (score: {pred.get('score', 'N/A')})"
        return data
    except Exception:
        return result



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    try:
        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = invoke_endpoint(features)
        return templates.TemplateResponse(
            "result.html", {"request": request, "prediction": prediction}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "result.html", {"request": request, "prediction": f"Error: {str(e)}"}
        )
