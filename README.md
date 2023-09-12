# Transformers as Support Vector Machines
This repository holds the official code for the paper [Transformers as Support Vector Machines](https://arxiv.org/abs/2308.16898)


### Experimental Details
We create a 1-layer self-attention using PyTorch, training it with the SGD optimizer and a learning rate of $\eta=0.1$. We apply normalized gradient descent to ensure divergence of attention weights. The attention weight $W$ is then updated through

$$W(k+1)=W(k)-\eta\frac{\nabla\mathcal{L}(W(k))}{||\nabla\mathcal{L}(W(k))||_F}.$$
 
In the setting of $(K,Q)$-parameterization, we noted that with extended training iterations, the norm of the combined parameter $KQ^\top$ consistently rises, despite the gradient being treated as zero due to computational limitations. To tackle this issue, we introduce a minor regularization penalty to the loss function, ensuring that the norms of $K$ and $Q$ remain within reasonable bounds. This adjustment involves

$$\widetilde{\mathcal{L}}(K,Q)=\mathcal{L}(K,Q)+\lambda(||K||^2_F+||Q||^2_F).$$

Here, we set $\lambda$ to be the the smallest representable number, e.g. computed as $1+\lambda=1$ in Python, which is around $2.22\times10^{-16}$. Therefore, $K,Q$ parameters are updated as follows.

$$K(k+1)=K(k)-\eta\frac{\nabla\widetilde{\mathcal{L}}_K(K(k),Q(k))}{||\nabla\widetilde{\mathcal{L}}_K(K(k),Q(k))||_F}, \qquad Q(k+1)=Q(k)-\eta\frac{\nabla\widetilde{\mathcal{L}}_Q(K(k),Q(k))}{||\nabla\widetilde{\mathcal{L}}_Q(K(k),Q(k))||_F}.$$

Additionally, to evaluate the similarity of two matrices, we use the following measurement:

$$\text{Correlation coefficient}(W_1,W_2)=\frac{\langle W_1,W_2\rangle}{||W_1||_F||W_2||_F}.$$



### Requirements

```
torch
cvxpy
```

### Reproducing Results 

- Global convergence:

  - *visualization.ipynb*: Visualization of GD paths and SVM directions for training $W$ (solid) or $(K,Q)$ (dashed) parameters.

- Local convergence:

  - *local_converge.ipynb*: Evolutions of softmax probabilities when training $W$ (blue) or $(K,Q)$ (red) parameters and correlation coefficients to $W_\alpha^{\text{mm}}$ (blue) or $W^{\text{mm}}_{\star,\alpha}$ (red).

- Low rank experiments: 

  - *svm_rank.ipynb*: Investigation of rank of SVM solutions with Frobenius norm (solid) or nuclear norm (dashed) under different $(n,T,d)$ choices.
