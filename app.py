from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import io
from PIL import Image

app = FastAPI()

# Load trained model
model = tf.keras.models.load_model("cat_dog_model.h5")

@app.get("/")
def home():
    return {"message": "Cat-Dog Classifier is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((128, 128))  # same size as training

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"

    return JSONResponse(content={"prediction": result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
