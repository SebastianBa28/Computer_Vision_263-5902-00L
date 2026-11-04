# Computer Vision HS2026 Lab 3

Deep Learning for Computer Vision
TA: Dingxi Zhang, zhangdi@ethz.ch
Submission date: 6 November 2026. Moodle specifies the closing time.

## Setup

From this folder, run:

```bash
conda env create -f env.yml
conda activate cv-lab3
jupyter notebook
```

Start with `pcl_binary_classification.ipynb`, then continue with
`image_classification.ipynb`. Complete the TODOs before running the corresponding
training cells. Keep the supplied function interfaces unchanged.

The included environment uses CPU PyTorch. For the optional GPU bonus, use a
GPU-enabled PyTorch environment or a Colab GPU runtime, and confirm that
`torch.cuda.is_available()` is true. MNIST, CIFAR-10, and pretrained ImageNet
weights are downloaded when first used; an internet connection is required.
On Colab, also upload `data/2d_pcl_dataset.npz` if running the point-cloud notebook.

If a data loader fails, try `NUM_WORKERS = 0` to see a clearer error.
The first MNIST CNN run on CIFAR-10 is intentionally expected to fail with an
input-channel error. Then complete the CIFAR adaptation and rerun its training.

Submit your completed code in one ZIP through Moodle. No report is required.
