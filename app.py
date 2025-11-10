from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np

app = FastAPI()

# ✅ Allow CORS (important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # agar chaho to specific frontend URL bhi de sakte ho
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("cat_dog_model.h5")

@app.post("/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).resize((128, 128))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)
    result = "Dog" if prediction[0][0] > 0.5 else "Cat"
    return {"prediction": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

