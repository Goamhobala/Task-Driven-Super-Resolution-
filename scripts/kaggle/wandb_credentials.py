import os
import wandb
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")

os.environ["WANDB_API_KEY"] = wandb_api_key
wandb.login(key=wandb_api_key)