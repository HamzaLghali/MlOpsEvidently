from typing import Dict
from kserve import Model, ModelServer, InferRequest, InferResponse
from kserve.utils.utils import get_predict_input, generate_uuid
import os

class MyCustomModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()
    
    def load(self):
        # TODO: Implement your model loading logic here
        # e.g., self.model = joblib.load('model.pkl')
        print(f"Model {self.name} loaded.")
        self.ready = True
        
    def predict(self, payload: InferRequest, headers: Dict[str, str] = None) -> InferResponse:
        # TODO: Implement your prediction logic
        # 1. Extract inputs from payload.inputs or payload (if raw request)
        #    raw_data = payload # if using raw request bytes
        #    deserialized_data = self.decode_request(payload) # if using kserve.InferInput
        #    input_data = [infer_input.data for infer_input in payload.inputs if infer_input.name == "your_input_name"]
        # Also with InferRequest: input_features = get_predict_input(payload)
        
        # 2. Perform inference using self.model
        #    predictions = self.model.predict(processed_input)
        
        # 3. Format the output as kserve.InferResponse
        #    infer_output = kserve.InferOutput(name="output-0", shape=list(predictions.shape), datatype="FP32", data=predictions.tolist())
        #    return InferResponse(model_name=self.name, infer_outputs=[infer_output])
        
        # Example for a simple pass-through or echo
        print(f"Received payload: {payload}")
        return InferResponse(model_name=self.name, response=payload, infer_outputs=[])  # Adjust as needed
    
    # Optional: Implement preprocess
    # def preprocess(self, payload: InferRequest, headers: Dict[str, str] = None) -> InferRequest:
    #     # TODO: Implement preprocessing
    #     print(f"Preprocessing payload: {payload}")
    #     return payload
    
    # Optional: Implement postprocess
    # def postprocess(self, response: InferResponse, headers: Dict[str, str] = None) -> InferResponse:
    #     # TODO: Implement postprocessing
    #     print(f"Postprocessing response: {response}")
    #     return response

if __name__ == "__main__":
    # Get model name from environment variable or use default
    model_name = os.getenv('MODEL_NAME', 'churn-predictor')
    model = MyCustomModel(model_name)
    ModelServer().start([model])