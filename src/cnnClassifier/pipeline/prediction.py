import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import dlib
from src.cnnClassifier.components.data_preprocessing import DataPreprocessing
from src.cnnClassifier.config.configuration import ConfigurationManager
import cv2
class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Initialize ConfigurationManager and fetch configurations
        self.config_manager = ConfigurationManager()
        self.data_preprocessing_config = self.config_manager.get_preprocess_data()
        self.data_preprocessor = DataPreprocessing(self.data_preprocessing_config)
        self.data_preprocessor = DataPreprocessing(self.data_preprocessing_config)
        self.faceDetector = dlib.get_frontal_face_detector()
        
    def preprocess_image(self, test_image):
        test_image_array = np.array(test_image)
        faces = self.faceDetector(test_image_array, 0)
        if len(faces) == 0:
            print(f"No faces found in image")
            return
        
        if faces:
            x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
            face_image_rgb = test_image_array[y:y+h, x:x+w]
            
        else:
            # If no face is detected, use the entire image
            face_image_rgb = test_image_array
        resized_face = cv2.resize(face_image_rgb, (512, 512))
        # face_image_array = np.array(resized_face)
        image_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        processed_image = np.zeros((image_rgb.shape), np.int16)
        # test_image_array = np.array(test_image)
        for i in range(3):
          processed_image[:,:,i] = self.data_preprocessor.ED_LBP_Sliding_Matrix(image_rgb[:,:,i].astype(np.int16), P=8)
        processed_image_dir = "artifacts/data_preprocessing"
        os.makedirs(processed_image_dir, exist_ok=True)
        
        # Construct the full path for the processed image
        # Extract the base filename without extension and add a suffix
        base_filename = os.path.splitext(os.path.basename(self.filename))[0] + "_processed.png"
        processed_image_path = os.path.join(processed_image_dir, base_filename)
        
        # Save the preprocessed image
        cv2.imwrite(processed_image_path, processed_image)
        return processed_image
    def predict(self):
            # load model
            model = load_model(os.path.join("artifacts","training", "best_model_v2.h5"))

            imagename = self.filename
            test_image = image.load_img(imagename, target_size = (512,512))
            test_image = self.preprocess_image(test_image)
            if test_image is None:
                print("Preprocessing failed; no prediction made")
                return None
            test_image = image.img_to_array(test_image)
            test_image = cv2.resize(test_image, (224, 224))
            test_image = test_image / 255.0
            test_image = np.expand_dims(test_image, axis = 0)
            prpability = model.predict(test_image)
            return prpability
            
            # if prpability >= 0.75:
            #     prediction = 'Real'
            #     return [{ "image" : str(prpability),"Statue": prediction}]
            # else:
            #     prediction = 'fake'
            #     return [{ "image" : str(prpability), "Statue": prediction}]