
# Classifier calibration example

Training classifiers with log loss (cross entropy) is prone to overfitting because
the predictions are conditioned on the feature vectors. This overfitting manifests by
having over-optimistic or -pessimistic probability scores even when classification
accuracy is high.

Here's an example of training a well-calibrated classifier network by using
regularization methods like
* focal loss,
* label smoothing,
* temperature scaling, or
* weight decay, and
conducting a hyperparameter search to optimize held-out calibration error.

The code is loosely based on the original [PyTorch MNIST example](https://github.com/pytorch/examples/tree/master/mnist).

## Running

Create a Conda environment and run the script:

    conda env create -n mnist_calibration -f environment.yml
    conda activate mnist_calibration
    python mnist_calibration.py

## References

* [Calibrating Deep Neural Networks using Focal Loss](https://arxiv.org/abs/2002.09437)
* [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
* [Verified Uncertainty Calibration](https://arxiv.org/abs/1909.10155)
* [When Does Label Smoothing Help?](http://www.cs.toronto.edu/~hinton/absps/smoothing.pdf)
* [Uncertainty Calibration Library](https://github.com/p-lambda/verified_calibration)
