# PTAflow
Neural posterior estimation for Pulsar Timing Arrays

Code to reproduce the results in https://arxiv.org/abs/2310.12209

D. Shih, M. Freytsis, S. R. Taylor, J. A. Dror and N. Smyth,
"Fast Parameter Inference on Pulsar Timing Arrays with Normalizing Flows,"
[arXiv:2310.12209 [astro-ph.IM]].

Requires pytorch, nflows.

Download the training data from here:

https://zenodo.org/doi/10.5281/zenodo.10906129

Train model for 100 epochs:

python train_model.py

Prints out val loss and saves each epoch. 

Currently set up to train on 3 GPUs with DataParallel. Takes approximately 15 minutes / epoch on 3 Tesla P100 or 3 GeForce GTX 1080 Ti cards. 

