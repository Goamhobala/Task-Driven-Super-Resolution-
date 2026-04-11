# InstaRoad Prototype
The task is to create a prototype for road detection using Sentinel 2 data. The work done here can be used to assist in creating the proposol. This would be a shared central repository containing all our work and would link all the forks and submodules.

Organisation is important. Collaboration can be messy. It is a good idea to follow some good practices:
- Descriptive commit messages of changes
- Main branch contains code that is ready to be run (experiments, inference...)
- Other branches contains feature specific changes, which we would then merge into main
- Documentation is created via Markdown files (docs folder) and served through the web browser with `mkdocs`
  - Can docs can be served locally with `mkdocs serve`
  - Future plans to serve it via cloudflare

## Goals
A few directions that I think we could explore. Potentially also use a seperate scheduling application (Trello, ToDoist, Clickup) to help out. The goals are currently roughly structured from basic to more advanced.

I think it's currently feasible to try get a basic pixel based model (Unet baseline and something better) trained on our own small dataset.

### Models
1. Pixel-Based Road Detection Models
2. Graph-Based Models
3. Geofoundational Models
4. Super resolution models
5. Self-Supervised Learning Models
6. Temporal Models

### Dataset
1. Data preprocessing on Kaggle Indian Road dataset
2. Creation of own urban dataset
   1. Location?: City of Cape Town
   2. Bands: RGB, multispectral, SAR
   3. Road Vector Option: road label with S2 Data
   4. Different resolutions and scales. To try have model extract what makes a road a road. Potentially help with scale differences of highways and narrow roads.
   5. High resolution option: High resolution downsample with high quality road labels
   6. Differentation of different road-like features: Rivers, Canals, overground gas pipelines, railroads.
3. Creation of own rural dataset. South African rural regions
4. High resolution and overlapping road
5. Self-supervised / unsupervised techniques

Metrics evaluation metrics should try to be consistent across datasets. Precision, Recall, IoU.

### Inference
1. Basic inference batch file output of predictions for visualisation
2. Overlay predictions on some sort of map (raw tiff files are kinda resource intensive - maybe something better to overlay many predictions)
3. Integration with InstaGeo Framework

## Some General Thoughts
Realised we don't communicate and collaborate enough. Seriously, if we want to do well, we need to all contribute, help and build on top of each other's ideas. Preferabily play to our strengths.

We probably should also have tasks assigned to every one each week. Again probably a scheduling program. I like Trello or ToDoist. Would also help with distributing workload of proposol.

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

> More documentation is served via mkdocs on the web. Or can be found in docs folder