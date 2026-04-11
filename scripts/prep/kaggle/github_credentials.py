import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ['GIT_TOKEN'] = user_secrets.get_secret("Github_kaggle")

os.environ['GIT_USER'] = "instaroad"
os.environ['REPO_NAME'] = "InstaRoad"