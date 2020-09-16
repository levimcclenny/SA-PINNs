## Self-Adaptive PINN - Official Implementation



### Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism
#### Levi McClenny<sup>1,2</sup>, Ulisses Braga-Neto<sup>1</sup>

Paper: https://arxiv.org/pdf/2009.04544.pdf

Abstract: *Physics-Informed Neural Networks (PINNs) have emerged recently as a promising application of deep neural networks to the numerical solution of nonlinear partial differential equations (PDEs).
However, the original PINN algorithm is known to suffer from stability and accuracy problems in
cases where the solution has sharp spatio-temporal transitions. These ``stiff'' PDEs require an unreasonably large number of collocation points to be solved accurately. It has been recognized that adaptive procedures are needed to force the neural network to fit accurately the stubborn spots in the solution of stiff PDEs. To accomplish this, previous approaches have used fixed weights hard-coded over regions of the solution deemed to be important. In this paper, we propose a fundamentally new method to train PINNs adaptively, where the adaptation weights are fully trainable, so the neural network learns by itself which regions of the solution are difficult and is forced to focus on them, which is reminiscent of soft multiplicative-mask attention mechanism used in computer vision. The basic idea behind these Self-Adaptive PINNs is to make the weights increase where the corresponding loss is higher, which is accomplished by training the network to simultaneously minimize the losses and maximize the weights, i.e., to find a saddle point in the cost surface. We show that this is formally equivalent to solving a PDE-constrained optimization problem using a penalty-based method, though in a way where the monotonically-nondecreasing penalty coefficients are trainable.
Numerical experiments with an Allen-Cahn ``stiff'' PDE, the Self-Adaptive PINN outperformed other state-of-the-art PINN algorithms in L2 error by a wide margin, while using a smaller number of training epochs. An Appendix contains additional results with Burger's and Helmholtz PDEs, which confirmed the trends observed in the Allen-Cahn experiments.*

<sub><sub><sup>1</sup>Texas A&M Dept. of Electrical Engineering, College Station, TX</sub></sub><br>
<sub><sub><sup>2</sup>US Army CCDC Army Research Lab, Aberdeen Proving Ground/Adelphi, MD</sub></sub><br>

### Estimator Architecture


## Requirements
Code was implemented in ```python 3.7``` with the following package versions:
```
tensorflow version = 2.3
keras version = 2.2.4
```

and ```matplotlib 3.1.1``` was used for visualization.

### Virtual Environment (Optional)
**(Mac)** To create a virtual environment to run this code, download the repository either via ```git clone``` or by clicking download at the top of github, then navigate to the top-level folder in a terminal window and execute the commands

```
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
```

This will create a virtual environment named ```venv``` in that directory (first line) and drop you into it (second line). At that point you can install/uninstall package versions without effecting your overall environment. You can verify you're in the virtual environment if you see ```(venv)``` at the beginning of your terminal line. At this point you can install the exact versions of the packages listed here with the pip into the venv:

```
pip install tensorflow==2.3 numpy==1.15.4 keras==2.2.4
```

run
```
python versiontest.py
```

And you should see the following output:
```
Using TensorFlow backend
tensorflow version = 2.3
keras version = 2.2.4
numpy version = 1.15.4
```

## Data
The data used in this paper is publicly available in the Raissi implementation of Physics-Informed Neural Networks [found here](https://github.com/maziarraissi/PINNs). It has already been copied into the appropriate directories for utilization in the script files.

## Usage
You can recreate the results of the paper by simply navigating to the desired system (i.e. opening the Burgers folder) and running the .py script in the folder. After opening the Burgers folder, simply run
```
python burgers.py
```

And the training will begin, followed by the plots.

## Note

The results in the paper were calculated on GPU. Running for the full 10k/10k training iterations for Adam and L-BFGS will likely take a very long time on CPU.

## Citation
Cite using the Bibtex citation below:

```
@article{mcclenny2020self,
  title={Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism},
  author={McClenny, Levi and Braga-Neto, Ulisses},
  journal={arXiv preprint arXiv:2009.04544},
  year={2020}
}

```
