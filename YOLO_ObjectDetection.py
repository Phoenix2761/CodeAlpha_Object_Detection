from ultralytics import YOLO

# Importing the keyboard module to record keystrokes
import keyboard

# Creating an instance of the yolov8x model
model = YOLO("yolov8x.pt")

# Training the yolov8x model on the coco8.yaml dataset with 100 epochs
training_data = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Validating the models accuracy using it's training dataset (The model.val() function takes the validation 
# dataset as an argument, since no dataset was provided, the validation function defaults to using the training
# dataset for validation.
model_metrics = model.val()
model_metrics.box.map
model_metrics.box.map50
model_metrics.box.map75
model_metrics.box.maps

# Specifying the built-in webcam as the value of the source parameter
# Giving the "show" parameter a value of true so that we can see the object in realtime as the model detects it 
# Giving the model a confidence of 0.4
# Giving the "save" parameter a value of "True" so that the results of the detection can be saved
# Specifying "True" as the value of the "stream" parameter so as to utilize a memory efficient generator 
# thereby storing only the results of the current frame or datapoint hence saving memory.
model_responses = model(source=0, show=True, conf=0.4, save=True, stream=True)

# Looping through each response in the result of the models predictions, since stream=True only the current 
# frame/datapoint will be saved to memory.
# Showing/displaying the frames in realtime as the model is detecting the object and 
# Saving the models prediction to the default storage location (runs\detect)
for model_response in model_responses:
    model_response.save(filename="DetectedObjects.jpg")
    # Stopping the detection when the key "q" is pressed by breaking the for loop
    if keyboard.is_pressed("q") == True:
        break