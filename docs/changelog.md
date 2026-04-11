# ChangeLog

## 11 April 2026 (Kelvin)
Mostly importing changes from old repository to new repository. At the same time, structuring the repository to be better organised. Because it's mostly importanting, the code isn't refactored. The documentation is also just written, hope to iteratively improve upon the documentation and code.

Biggest change is probably the usage of submodules. It is a bit of a pain, since it requires some extra commands to manage, but I believe that it'll be more maintainable.

### SAMRoad2 Repo Changes

#### Dataset Loading
Changed code to use lazy loading. This is because all the data for sentinel2 can't fit into the RAM.

It was found that the loaders treats the "test set" as a validation set and the "train + validation set" as the train set. Hence code was changed to actually use a test set. However, changes were not fully made for the other datasets (cityscale and spacenet).

#### Training Process
In the model class, (+  1e-6) to denominators of val loss calculations due to N/A values (masks with no road labels). We might need to find a way to better handle this.

The progress output was supressed as kaggle logs don't store those logs well. Kaggle logs become very hard to read due to how long it becomes.

#### Inference Process
Changes to use sentinel 2 dataset. Also added code create comparison output images.

#### Misc
Added configs and dataset config files
