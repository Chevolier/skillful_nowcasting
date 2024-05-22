FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.2.0-gpu-py310-cu118-ubuntu20.04-sagemaker

WORKDIR /app

RUN git clone https://github.com/openclimatefix/skillful_nowcasting.git
 
COPY . .

RUN pip install -r requirements.txt
RUN pip install -e skillful_nowcasting
RUN pip install pytest
