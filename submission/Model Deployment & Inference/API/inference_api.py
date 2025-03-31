from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import onnxruntime as ort
import os
import uvicorn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI()

#load all ONNX models from the directory they are saved at
model_directory = "C:\\Users\\PC\\OneDrive - Lebanese American University\\inmind\\ML track\\final project\\submission\\Model Deployment & Inference\\Netron"
model_sessions = {
    model_name: ort.InferenceSession(os.path.join(model_directory, model_name), providers=["CPUExecutionProvider"]) #specify using CPU
    for model_name in os.listdir(model_directory) if model_name.endswith(".onnx")
}

# Preprocessing transform
transform = A.Compose([
    A.Resize(640, 640), #resize the image to 640,640 as the yolo model was trained
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0), #normalize the pixel values
    ToTensorV2(),
])

@app.get("/", include_in_schema=False) #added just to show a message when clicking the link on run
def root():
    return {"message": "YOLOv5 ONNX Inference API is running, check the docs page and execute"}

@app.get("/models") #endpoint holding the names of trained yolo models
def list_models():
    return {"models": list(model_sessions.keys())}

@app.post("/bbox") #endpoint that allows doing inference for an image, then shows the bboxes in JSON format
async def detect_bbox(file: UploadFile = File(...), model_name: str = Form(...)):

    if model_name not in model_sessions: #make sure to use an existing model
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    session = model_sessions[model_name]

    #load and preprocess image
    image_pil = Image.open(file.file).convert("RGB")
    image_np = np.array(image_pil)
    orig_h, orig_w = image_np.shape[:2]  #get original size

    image = transform(image=image_np)["image"]
    input_tensor = np.expand_dims(image, axis=0)

    #run inference
    outputs = session.run(None, {"images": input_tensor})
    predictions = outputs[0][0]  #the output shape here is same as netron (25200, 10)

    results = []
    for pred in predictions:
        #fix the coordinates of the bounding box
        cx, cy, w, h = pred[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        objectness = pred[4]
        class_scores = pred[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = class_scores[class_id]
        confidence = objectness * class_conf

        if confidence > 0.5: #filter bboxes with low confidence
            # Scale to original size
            x1 = int(x1 / 640 * orig_w)
            y1 = int(y1 / 640 * orig_h)
            x2 = int(x2 / 640 * orig_w)
            y2 = int(y2 / 640 * orig_h)

            results.append({
                "class": class_id,
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    print("Predictions count:", len(results))
    print("First box:", results[0] if results else "No boxes")

    return JSONResponse(content={"model": model_name, "predictions": results}) #show them in JSON format


@app.post("/bbox-image") #endpoint displaying the image with its predicted bounding boxes
async def detect_bbox_image(file: UploadFile = File(...), model_name: str = Form(...)):
    if model_name not in model_sessions:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    session = model_sessions[model_name]
    
    image_pil = Image.open(file.file).convert("RGB")
    image_np_original = np.array(image_pil)  # shape: (H, W, 3)
    orig_h, orig_w = image_np_original.shape[:2]

    #convert from RGB to BGR
    image_np_original = cv2.cvtColor(image_np_original, cv2.COLOR_RGB2BGR)

    image_np_for_model = transform(image=image_np_original)["image"]
    input_tensor = np.expand_dims(image_np_for_model, axis=0)

    #run inference
    outputs = session.run(None, {"images": input_tensor})
    predictions = outputs[0][0]

    results = []
    for pred in predictions:
        cx, cy, w, h = pred[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        objectness = pred[4]
        class_scores = pred[5:]

        class_id = int(np.argmax(class_scores))
        class_conf = class_scores[class_id]
        confidence = objectness * class_conf

        if confidence > 0.5:
            x1 = int(x1 / 640 * orig_w)
            y1 = int(y1 / 640 * orig_h)
            x2 = int(x2 / 640 * orig_w)
            y2 = int(y2 / 640 * orig_h)
            results.append((x1, y1, x2, y2, confidence, class_id))

    #draw boxes
    for (x1, y1, x2, y2, conf, class_id) in sorted(results, key=lambda x: x[4], reverse=True)[:20]:
        label = f"{class_id}: {conf:.2f}"
        cv2.rectangle(image_np_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np_original, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #return image
    _, img_encoded = cv2.imencode('.png', image_np_original)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)