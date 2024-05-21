import torch
from dgmr import DGMR
import json

def model_fn(model_dir):
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

        print("request_body_json", request_body_json)

        return request_body_json

    # If the request_content_type is not as expected, raise an exception
    raise ValueError(f"Content type {request_content_type} is not supported")

def predict_fn(data, model):
    data = json.loads(data)
    
    forecast_steps = data.get("forecast_steps", 18)
    
    model.config['forecast_steps'] = forecast_steps
    model.sampler.forecast_steps = forecast_steps
    model.latent_stack.shape = data.get("latent_stack_shape", (8, 256//32, 256//32))
    
    input_frames = data.get("input_frames")
    input_frames = torch.tensor(input_frames).unsqueeze(0).unsqueeze(2)

    with torch.no_grad():
        pred_frames = model(input_frames.cuda())
        pred_frames[pred_frames<0] = 0
    
    results = {"pred_frames": pred_frames.cpu(), "forecast_steps": forecast_steps}
    
    return results