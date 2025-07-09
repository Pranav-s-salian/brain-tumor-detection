import cv2
import numpy as np

def load_and_predict(model_path, image_path):
    """
    Load saved model and make prediction on new image
    
    Args:
        model_path: Path to saved model (.h5 file)
        image_path: Path to image to predict
    
    Returns:
        prediction: 0 (no tumor) or 1 (tumor)
        confidence: confidence score
    """
    from tensorflow.keras.models import load_model
    
    
    loaded_model = load_model(model_path)
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0) 
    
    
    prediction = loaded_model.predict(img)[0][0]
    
    if prediction > 0.5:
        return 1, prediction  #  there is an tumor i
    else:
        return 0, 1-prediction  # No there is no tumor
    

prediction, confidence = load_and_predict('brain_tumor_model.h5', "path to the imaeg with or without tumor")
print(f"Result: {'Tumor Detected' if prediction == 1 else 'No Tumor'}")
print(f"Confidence: {confidence:.2f}")



