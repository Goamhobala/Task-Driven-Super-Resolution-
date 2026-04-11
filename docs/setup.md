# Setup

## Environment Setup
Python dependencies are handled with UV. Optional dependencies are used so only the dependencies that are needed are installed.

This repository uses submodules hence you have to git clone with recursive flag:
``` bash
git clone --recursive https://github.com/InstaRoad/InstaRoadPrototype.git
```

You can install and pick optional dependencies to be installed as follows:
``` bash
# adding optional dependencies with exact lock file
uv sync --locked --extra samroad --extra sentinel2

# adding package to a dependency
uv add numpy --optional samroad
```

You can also pip install them with:
``` bash
uv pip install -e ".[samroad,sentinel2]
```
> the system tag (--system) is typically used in notebook environments like kaggle and colab. So that the dependencies are installed systemwide and not just isolated in the specific virtual environment


| Optional Dependencies | Description                                  |
| --------------------- | -------------------------------------------- |
| samroad               | Samroad model                                |
| sentinel2             | Manipulation of sentinel2 data (in progress) |
| instageo              | Run instageo framework                       |
| terra                 | Terratorch and Terramind (in progress)       |
| utitiles              | Utilities shared (in progress)               |


## Remote Repo Setup
Training the models are computationally expensive. Hence, we take advantage of Kaggle and Google Colab for prototyping.
HCPC would be used for longer training runs.

Cloning the repo requires a Github PAT. This can be obtained from "Settings -> Developer Settings (bottom left) -> Personal Access Tokens -> Fine-grained tokens.

Click "Generate new Token" and fill-in the details of the token:
- Change Resource Owner to "InstaRoad"

Click on "Add Permissions"
- Check "Contents"

Now save this PAT token in Kaggle:
- Open a notebook -> Add-ons -> Secrets

The below code is useful for cloning the repo (colab):
``` python
import getpass
import os
from google.colab import userdata

user = "instaroad"
password = userdata.get('Github_kaggle')
repo_name = "InstaRoad"
cmd_string = f"git clone https://oauth2:{password}@github.com/{user}/{repo_name}.git"

!rm -rf /content/InstaRoad
!{cmd_string}
```

cloneing the repo (kaggle)
``` python
from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
github_pat = user_secrets.get_secret("InstaRoad_GITHUB_PAT")
github_user = "instaroad"
repo_name = "InstaRoad"

cmd_string = f"git clone --recursive https://oauth2:{github_pat}@github.com/{github_user}/{repo_name}.git"
%cd /kaggle/working
!rm -rf /content/InstaRoad
!git config --global url."https://github.com/".insteadOf git@github.com:
!{cmd_string}
```

## Kaggle Authentication Setup
This if datasets are generated on colab or private models/datasets to be downloaded. First save your kaggle username and kaggle key in to google colab secrets.

``` python
import os
from google.colab import userdata

# Set your Kaggle credentials
os.environ['KAGGLE_USERNAME'] = userdata.get('Kaggle_username')
os.environ['KAGGLE_API_TOKEN'] = userdata.get('Kaggle_Key')

# Test the connection
!kaggle competitions list
```

## Submodule Setup

### Adding submodules
``` bash
git submodule add https://github.com/username/repository-name.git path/to/destination_folder

# Checkout the submodule branch to prevent detached heads
cd path/to/destination_folder
git checkout main
# Also get it's submodules if they have
git submodule update --init --recursive
```

Committing changes in a submodule requires you to:

- make commit to the submodule
- make a commit to parent repo to update pointer to updated commit

Pulling updates:
```
git submodule update --remote path/to/destination_folder
```

### Changing submodules
Changing submodule repos (might need to do for sam road repo).
We want to change the `.gitmodules` files in `.git` folder.
```
[submodule "path/to/submodule1"]
    path = path/to/submodule1
    url = https://github.com/ORIGINAL-AUTHOR/submodule1.git
```

Then we commit:

```bash
# Update Git's internal configuration with your new URLs
git submodule sync

# Initialize and clone the submodules from YOUR forks
git submodule update --init --recursive

git add .gitmodules
git commit -m "Update submodules to point to personal forks"
git push origin main
```

## Weights and Bias Setup

### Kaggle Weights and Bias Setup

```python
import os
import wandb
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")

os.environ["WANDB_API_KEY"] = wandb_api_key
wandb.login(key=wandb_api_key)
```