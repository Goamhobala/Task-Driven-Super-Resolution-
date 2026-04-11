import os
from google.colab import userdata

# Fetch the secret and set environment variables
os.environ['GIT_TOKEN'] = userdata.get('Github_kaggle')
os.environ['GIT_USER'] = "instaroad"
os.environ['REPO_NAME'] = "InstaRoad"