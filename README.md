# Accelerating Triangle Counting in a Graph using GPU


## Requirement
The current version of code was tested on Cuda V10.2.89 and gcc version 4.8.5.

The baseline comparsion is done by turicreate (https://github.com/apple/turicreate)

```
 pip install -U turicreate

```

## Datasets used for testing

1. https://snap.stanford.edu/data/as-Skitter.html
2. https://snap.stanford.edu/data/com-Orkut.html
3. https://snap.stanford.edu/data/soc-LiveJournal1.htm

## Usage

Build the code
```
make
```
Custom testing mode
```
./parallelTriangleCount custom
```
Test with dataset (dataset needs to be filtered by removing self-loops and duplicate edges)
```
./parallelTriangleCount file ./dataset-filtered
```
