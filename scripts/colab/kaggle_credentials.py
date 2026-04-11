import os
from google.colab import userdata

# Set your Kaggle credentials
os.environ['KAGGLE_USERNAME'] = userdata.get('Kaggle_username')
os.environ['KAGGLE_API_TOKEN'] = userdata.get('Kaggle_Key')