Dataset link : https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

pip install kagglehub
import kagglehub

# Download latest version
path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")

print("Path to dataset files:", path)
