FROM nvcr.io/nvidia/pytorch:25.09-py3

RUN pip install transformers==4.57.0
RUN pip install accelerate==1.10.1
RUN pip install pandas
RUN pip install pyarrow==21.0.0
RUN pip install --no-deps liger-kernel==0.6.2
RUN CUDA_VISIBLE_DEVICES='' pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install --no-deps git+https://github.com/tascj/offload_adam.git@1d0efb1d925f9a703cf66b4fbafaf904de894ebc