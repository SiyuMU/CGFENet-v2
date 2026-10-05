# CGFENet-v2

Predict left-ventricular pressure and wall displacement from a ventricular mesh.

Use Python 3.8. Run the commands below from this folder.

## Install

```bash
python3 -m pip install -r requirements.txt "vtk==9.2.5"
```

## Example

```bash
python3 example/example.py --device cpu
```

Uses the included mesh and pretrained model. Results are saved in `example/output/`.

## Train

```bash
python3 train.py --device cuda:0
```

The dataset and training cache will be made publicly available after publication
of the paper. Download links will be added here.
