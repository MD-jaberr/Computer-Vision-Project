The aim of this project is to:
-	Augment images and save them with their labels to increase the number of data
-	Train a YOLOv5 model on training data and test the model
-	Tune the hyperparameters, train and test again
-	Inspect the performance of each train and test on Tensorboard
-	Evaluate the performance of the models
-	Export the model to ONNX
-	Use Netron to visualize the model
-	Create an inference API showing the bounding box x_center, y_center, width, height, class name, and confidence scores in JSON files

----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
**Why was the API created**
The aim of the api is to allow to use a saved version of the trained yolo model, make with it predictions and display some information regarding the generated predictions. In our case the we export the trained yolo model into ONNX since it will hold all the parameters of the model and we can use it to make inference. Thus, this ONNX version of the model is used in the api to make predictions and show the details we are interested in.

**How to launch the API:**
1. Run the python code inference_api.py
2. In the terminal a link appears as: _http://127.0.0.1:8000_ click on it, **or** type it manually on the browser
3. A message pops up saying that api has successfully openned, you should go to docs to see endpoints ==> **http://127.0.0.1:8000/docs**
4. At docs you can see three endpoints created, one shows the models, ones shows the bbox details, and the third shows the image with the detected bboxes

**Models Endpoint:**
Its role is to list the trained yolov5 models.
To see the trained yolov5 models follow the steps:
1. click on _"Try it out"_
2. click on _"Execute"_
3. the response body will show the available models

**Bounding Boxes Endpoint:**
Its role is to display the information about the bounding boxes, predicted on an image, with confidence more than a specified threshold in a JSON format.
To execute this endpoint do the following:
1. click on _"Try it out"_
2. In _"file"_ click on _"choose file"_ which will be the image
3. In _"model_name"_ write the name of the model you would like to use for bbox predictions
4. click on _"Execute"_
5. the response body will show the output holding: model name used with class_id of bbox, confidence of each bbox, coordinates of each bbox
6. The coordinates are as follows [left, top, right, bottom]

**Bounding Box Display Endpoint:**
Its role is to show the image and the predicted bboxes drawn on it.
To execute this endpoint do the following:
1. click on _"Try it out"_
2. In _"file"_ click on _"choose file"_ which will be the image
3. In _"model_name"_ write the name of the model you would like to use for bbox predictions
4. click on _"Execute"_
5. the image inserted will be displayed and the predicted bboxes will be drawn with their class_id and confidence both specified

**How to stop the API execution:**
The API's execution can be stopped through pressing _"Ctrl + C"_ on the terminal.

