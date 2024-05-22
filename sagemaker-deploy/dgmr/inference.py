import torch
from dgmr import DGMR
import json

def model_fn(model_dir):
    
    print("model_fn-----------------------")
    
    model = DGMR.from_pretrained(model_dir)
    model.eval()
    model.cuda()

    return model

def input_fn(request_body, request_content_type):
    """
    The request_body is passed in by SageMaker and the content type is passed in 
    via an HTTP header by the client (or caller).
    """

    print("input_fn-----------------------")

    if request_content_type == "application/json":
        request_body_json = json.loads(request_body)

        return request_body_json

    raise ValueError(f"Content type {request_content_type} is not supported")

def predict_fn(input_data, model):
    print("predict_fn---------------------")
    print(f"input_data type: {type(input_data)}")
    
    forecast_steps = input_data.get("forecast_steps", 18)
    
    model.config['forecast_steps'] = forecast_steps
    model.sampler.forecast_steps = forecast_steps
    model.latent_stack.shape = input_data.get("latent_stack_shape", (8, 256//32, 256//32))
    
    input_frames = input_data.get("input_frames")
    input_frames = torch.tensor(input_frames).unsqueeze(0).unsqueeze(2)
    
    with torch.no_grad():
        pred_frames = model(input_frames.cuda())
        pred_frames[pred_frames<0] = 0
    
    print(f"pred_frames shape: {pred_frames.shape}")
    pred_frames = pred_frames.cpu().squeeze()[0].tolist()
    results = {"pred_frames": pred_frames, "forecast_steps": forecast_steps}
    
    return results

def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    
    print("output_fn-----------------------")
    
    return json.dumps(predictions)