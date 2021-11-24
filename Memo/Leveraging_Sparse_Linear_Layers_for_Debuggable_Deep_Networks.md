# Leveraging Sparse Linear Layers for Debuggable Deep Networks

    最后一层的线性层用于可解释性！
    
    https://arxiv.org/pdf/2105.04857.pdf

[TOC]

## Abstract

We show how **fitting sparse linear models over learned deep feature representations can lead to more debuggable neural networks. These networks remain highly accurate while also being more amenable to human interpretation,** as we demonstrate quantitatively via numerical and human experiments. We further illustrate **how the resulting sparse explanations can help to identify spurious correlations, explain misclassifications, and diagnose model biases in vision and language tasks.**



## 1. Introduction

As machine learning (ML) models find wide-spread application, there is a growing demand for interpretability: access to tools that help people see why the model made its decision. **These obstacles stem from the scale of modern deep networks, as well as the complexity of eve4n defining and accessing the (often context-dependent) desiderata of interpretability.**

Existing work on deep network interpretability has largely approached this problem from two perspectives. **The first one seeks to uncover the concepts associated with specific neurons in the network**, for example through visualization or semantic labeling. **The second aims to explain model decisions on a pre-example basis, using techniques such as local surrogates and saliency maps.** While both families of approaches can improve model understanding at a local level -i.e., for a given example or neuron-**recent work has argued that such localized explanations can lead to misleading conclusions about the models' overall decision process.** As a result, it is often challenging to flag a model's failure modes or evaluate corrective interventions without in-depth problem-specific studies.

To make progress on this front, We focus on a more actionable intermediate goal of interpretability: ***model debugging*. Specifically, instead of directly aiming for a complete characterization of the model's decision process, our objective is to develop tools that help model designers uncover unexpected model behaviors (semi-) automatically.**



**Our contributions**. Our approach to model debugging is based on a natural view of a deep network as the composition of a "deep feature extractor" and a linear "decision layer". Embracing this perspective allows us to focus our attention on probing how deep features are (linearly) combined by the decision layer to make predictions. Even with this simplification, probing current deep networks can be intractable given the large number of parameters in their decision layer of a deep network with a sparse but comparably accurate counterpart. We find that this simple approach ends up being surprisingly effective for building deep networks that are intrinsically  more debuggable. Specifically, for a variety of modern ML settings:

+ We demonstrate that **it is possible to construct deep networks that have sparse decision layers (e.g., with only 20-30 deep features per class for ImageNet) without sacrificing much model performance.** This involves **developing a custom solver for fitting elastic net regularized linear models in order to perform effective sparsification  at deep-learning scales.**
+ We show that sparsifying a network's decision layer can indeed help humans understand the resulting models better. For example, untrained annotators can intuit (simulate) the predictions of a model with a sparse decision layer with high (~63%) accuracy. This is in contrast to their near chance performance (~33%) for models with standard (dense) decision layers.
+ We explore the use of sparse decision layers in three debugging tasks: diagnosing biases and spurious correlations, counterfactual generation, and identifying data patterns that cause misclassifications. To enable this analysis, we design a suite of human-in-the-loop experiments.



## 2. Debuggability via Sparse Linearity

We choose to decompose a deep network into: (1) a deep feature representation and (2) a linear decision layer.

allow us to get the best of both worlds: the predictive power of learned deep features, and the ease of understanding linear models.

we instead combine the feature representation of a pre-trained network with a sparse linear decision layer.

### 2.1 Constructing sparse decision layers

Let $(X,y)$ be the standardized data matrix (mean zero and variance one) and output respectively. In our setting, $X$ corresponds to the (normalized) deep feature representations of input data points, while $y$ is the target. Our goal is to fit a sparse linear model of the form $\mathbb{E}(\mathcal{Y} | X = x) = x^T \beta + \beta_0$. Then, the elastic net is the following convex optimization problem:
$$
\min _{\beta} \frac{1}{2N} \|X^T \beta + \beta_0 - y\|_2^2 + \lambda \mathcal{R}_{\alpha}(\beta)
$$
where
$$
\mathcal{R}_{\alpha}(\beta) = (1 - \alpha) \frac{1}{2} \|\beta\|_2^2 + \alpha \|\beta\|_1
$$
is referred to as the elastic net penalty for given hyperparameters $\lambda$ and $\alpha$. Typical elastic net solvers optimize for a variety of regulation strengths $\lambda_1 > \cdots > \lambda_k$, resulting in a series of linear classifiers with weights $\beta_1, \cdots, \beta_k$ known as the regularization path, where
$$
\beta_i = \operatorname{argmin} _{\beta} \frac{1}{2N} \|X^T\beta - y\|_2^2 + \lambda_i \mathcal{R}_{\alpha}(\beta)
$$
In particular, a path algorithm for the elastic net calculates the regularization path where sparsity ranges the entire spectrum from the trivial zero model ($\beta = 0$) to completely dense. This regularization path can then be used to select a single linear model to satisfy application-specific sparsity or accuracy thresholds (as measured on a validation set). 





## 3. Are Sparse Decision Layers Better?

We demonstrate that:

1. **The standard (henceforth referred to as "dense") linear decision layer can be made highly sparse at only a small cost to performance.**
2. **The deep features selected by sparse decision layers are qualitatively and quantitatively better at summarizing the models' decision process.**
3. **These aforementioned improvements (induced by the sparse decision layer) translate into better human understanding of the model.**



## 4. Debugging Deep Networks

**We now demonstrate how deep neural networks with sparse decision layers can be substantially easier to debug than their dense counterparts. We focus on three problems: detecting biases, creating counterfactuals, and identifying input patterns responsible for misclassifications.**



## 5. Related Work

### Interpretability Tools

### Regularized GLMs and Gradient Methods



## 6. Conclusion

We demonstrate **how fitting sparse linear models over deep representations can result in more debuggable models, and provide a diverse set of scenarios showcasing the usage of this technique in practice.** The simplicity of our approach allows it to be broadly applicable to any deep network **with a final linear layer, and may find uses beyond the language and vision settings considered in this paper.**

Furthermore, **we have created a number of human experiments for tasks such as testing model simulatability, detecting spurious correlations and validating misclassifications.** Although **primarily used in the context of evaluating the sparse decision layer, the design of these experiments may be of independent interest.**

Finally, we recognize that **while deep networks are popular within machine learning and artificial intelligence settings, linear models continue to be widely used in other scientific fields. We hope that the development and release of our elastic net solver will find broader use in the scientific community for fitting large scale linear models in contexts beyond deep learning.**

