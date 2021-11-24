# Towards the Unification and Robustness of Pertubation and Gradient Based Explanations

    LIME的理论解释，gradient based，最后会收敛到同一个地方。
    
    https://arxiv.org/pdf/2102.10618.pdf

[TOC]

## Abstract

As machine learning black boxes are increasingly being deployed in critical domains such as healthcare and criminal justice, there has been a growing emphasis on **developing techniques for explaining these black boxes in a post hoc manner.** In this work, **we analyze two popular post hoc interpretation techniques: SmoothGrad which is a gradient based method, and a variant of LIME which is a perturbation based method.** More specifically, **we derive explicit closed form expressions for the explanations output by these two methods and show that they both converge to the same explanation in expectation, i.e., when the number of perturbed samples used by these methods is large**. **We then leverage this connection to establish other desirable properties, such as robustness, for these techniques**. **We also derive finite sample complexity bounds for the number of perturbations required for these methods to converge to their expected explanation**. Finally, we empirically validate our theory using extensive experimentation on both synthetic and real world datasets.



## 1. Introduction

**Motivating the need for tools and techniques that can explain them in a faithful and human interpretable manner.**

Several techniques have been recently proposed to construct post hoc explanations of complex predictive models. While these techniques differ in a variety of ways, they can be broadly categorized into perturbation vs. gradient based techniques, based on the approaches they employ to generate explanations. For instance, LIME and SHAP (Ribeiro et al.,
2016; Lundberg & Lee, 2017) are called perturbation based methods because they leverage perturbations of individual instances to construct interpretable local approximations (e.g., linear models), which in turn serve as explanations of individual predictions of black box models. On the other
hand, SmoothGrad, Integrated Gradients and GradCAM (Simonyan
et al., 2014; Sundararajan et al., 2017; Selvaraju et al., 2017; Smilkov et al., 2017) are referred to as gradient based methods since they leverage gradients computed at individual instances to explain predictions of complex models.

**Explanations generated using perturbation based techniques such as LIME and SHAP may not be robust, i.e., the resulting explanations may change drastically with very small changes to the instances.**

**Gradient based methods such as Smooth-Grad and GradCAM may not generate interpretations that are faithful to the underlying models.**

**In this work, we initiate a study to unify perturbation and gradient based post hoc explanation techniques. To the best of our knowledge, this work makes the first attempt at establishing connections between these two popular classes of explanation techniques.**

+ We derive explicit closed form expressions for the explanations output by these methods and demonstrate that they converge to the same output (explanation) in expectation, i.e., when the number of perturbed samples used by these methods is large.

+ We then leverage this equivalence result to establish other desirable properties of these methods. More specifically, we prove that SmoothGrad and C-LIME satisfy Lipschitz continuity and are therefore robust to small changes in the input when the number of perturbed samples is large. This work is the first to demonstrate that a variant of LIME is provably robust.

+ We also derive finite sample complexity bounds for the number of perturbed samples required for SmoothGrad and C-LIME to converge to their expected output.

+ Finally, we prove that both SmoothGrad and C-LIME satisfy other interesting properties such as linearity.

  


## 2. Preliminaries

### 2.1 SmoothGrad

$$
\mathrm{SG}_{S}^{f}(x)=\frac{1}{|S|} \sum_{a \in S} \nabla f(a)
$$

$$
\mathrm{SG}_{P}^{f}(x)=\mathbb{E}_{a \sim P}[\nabla f(a)]
$$



### 2.2 LIME

$\pi: X \times X \rightarrow \mathbb{R}^{\geq 0}$ a distance metric over $X$. 
$$
\operatorname{LIME}_{S}^{f}(x)=\underset{g \in G}{\arg \min }\left\{L_{x}(f, g, S, \pi)+\Omega(g)\right\}
$$

where the loss function $L$ is defined as

$$
L_{x}(f, g, S, \pi)=\frac{1}{|S|} \sum_{a \in S} \pi(x, a)[f(a)-g(a)]^{2}
$$



### 2.3 Our Setting and Assumptions

$$
\text { C-LIME }_{S}^{f}(x)=\underset{g \in G}{\arg \min } \frac{1}{|S|} \sum_{a \in S}[f(a)-g(a)]^{2}
$$



## 3. Equivalence and Robustness

### 3.1 Equivalence

**We point out that Theorem 1 holds for any covariance matrix and does not require the covariance matrix to be diagonal. Furthermore, we note that the closed forms for both Smooth-Grad and C-LIME have a nice structure.**

For a diagonal $\Sigma$, the $i$’th coefficient of $\text{SmoothGrad}^f (x)$ and $\text{C-LIME}^f (x)$ depends only on the covariance of $f$ and the $i$'th feature. In particular, when $\Sigma = \sigma^2 \mathbf{I}$, then the $i$'th feature. In particular, when $\mathbf{\Sigma} = \sigma^2 \mathbf{I}$, then the $i$' coefficient is simply $\operatorname{cov}(f(a), a_i)\sigma^2$. This term captures the dependence of $f$ on the $i$'th feature of the input.



### 3.2 Robustness

**It is hence desirable to have robust explainability methods where two nearby points with similar labels have similar explanations.**

**Theorem 2**. Let $f: \mathbb{R}^d \rightarrow \mathbb{R}$ be a function whose gradient is bounded by $\nabla f_{max}$ and suppose $\Sigma = \sigma^2 \mathbf{I}$. Then $SG_{\Sigma}^f$ and $\text{C-LIME}_{\Sigma}^f$ are both $L$-Lipschitz with $L = \nabla f_{max} / (2 \sigma)$.

**Theorem 2 shows that both SmoothGrad and C-LIME be come**
**less robust (i.e., the Lipschitz constants grows) when explaining functions with larger magnitude of gradients, or when the variance parameter $\sigma^2$ used in gradient computation or perturbations decreases. However, the Lipschitz constant is independent of the input dimension $d$.**



## 4. Convergence Analysis

**The results in Section 3 prove the equivalence of Smooth-Grad and C-LIME and also robustness of these techniques in expectation which corresponds to large sample limits in practice.** Any useful implementation of these techniques is based on finite number of gradient computations or sample perturbations. **In this section, we derive sample complexity bounds to examine how fast the empirical estimates for the outputs of SmoothGrad and C-LIME at any given point will converge to the their expected value.** This extends the implications of the results in Section 3 to practical implementations of SmoothGrad and C-LIME.

**Proposition 1**. Let $f: \mathbb{R}^d \rightarrow \mathbb{R}$ be a function whose gradient is bounded by $\nabla f_{max}$. Fix $x \in X$, $\epsilon > 0$ and $\delta > 0$. Let $n \geq C(\nabla f_{max}/\epsilon)^2 \ln(d/\delta)$ for some absolute constant $C$. Then with probability of at least $1 - \delta$, over a sample $S$ of size $n$ from $\mathcal{N}(x, \Sigma)$, for any $\Sigma \in \mathbb{R}^d$, we have that $\|SG_{\Sigma}^f (x) - SG_{n}^f (x) \|_2 \leq \epsilon$.

We next examine how fast the output of C-LIME will converge
to its expectation.

**Theorem 3**. Let $f: \mathbb{R}^d \rightarrow [-1, 1]$ be a function. Fix $x \in X$, $\epsilon > 0$ and $\delta > 0$. Let $S$ denote a sample of size $n$ from $\mathcal{N} (x, \Sigma)$ for $\Sigma = \sigma^2 \mathbf{I}$ where
$$
n \geq C \frac{d \ln(\frac{d}{\delta})}{\min (\epsilon \delta^2, \epsilon \delta^3/\|x\|_2, \|x\|_2, \frac{1}{\sigma^2})^2}
$$
for some absolute constant $C$. Then with probability of at least $1 - \delta$, $\|\text{C-LIME}_{\Sigma}^f(x) - \text{C-LIME}_n^f(x) \|_2 \leq \epsilon$.



## 5. Additional Properties

The first property that we study is linearity.

**Proposition 2 (Linearity)**. Fix a covariance matrix $\Sigma \in \mathbb{R}^d \times \mathbb{R}^d$. For all $f, g$: $\mathbb{R}^d \rightarrow \mathbb{R}$, $d \in \mathbb{N}$, and $\alpha, \beta \in \mathbb{R}$
$$
SG_{\Sigma}^{\alpha f + \beta g} = \alpha SG_{\Sigma}^f + \beta SG_{\Sigma}^g, and
$$

$$
\text{C-LIME}_{\Sigma}^{\alpha f + \beta g} = \alpha \text{C-LIME}_{\Sigma}^f + \beta \text{C-LIME}_{\Sigma}^g
$$

Linearity implies that the explanation of a more complex function that can be written as a linear combination of two simpler functions is simply the linear combination of the explanations of each of the simpler functions.

The next property we study is proportionality.

**Proposition 3 (Proportionality)**. Let $f: \mathbb{R}^d \rightarrow \mathbb{R}$ be a linear function of the form $f(x) = \theta^T x + b$ for $\theta \in \mathbb{R}^d$ and $b \in \mathbb{R}$. For any $x \in \mathbb{R}^d$ and $\Sigma \in \mathbb{R}^d \times \mathbb{R}^d$
$$
SG_{\Sigma}^f (x) = \text{C-LIME}_{\Sigma}^f(x) = k(x) \theta
$$
for some function $k: \mathbb{R}^d \rightarrow \mathbb{R}$.

**Proportionality implies that when the underlying function is linear both SmoothGrad and LIME provide explanations that are proportional to the weights of the underlying function.** 

**Although explaining the weights of a linear function with another set of weights might appear unnecessary, proportionality can be interpreted as a sanity check for explain ability methods.**

**An immediate consequence of proportionality is that, in general, SmoothGrad and C-LIME do not provide sparse explanations (for e.g., when the underlying function is f linear and non-sparse).**



## 6. Experiments



## 7. Relate Work

Interpretability research can be categorized into learning inherently interpretable models, and constructing post hoc explanations. We provide an overview below.

### Inherently Interpretable Models

### Post Hoc Explanations

### Analyzing Post Hoc Explanations



## 8. Future Work

We initiate a study on **the unification of perturbation and gradient based post hoc explanations, and pave the way for several promising research directions.** It would be interesting to **establish connections between other perturbation and gradient based explanations such as SHAP or Integrated gradients.** It would also be interesting to **study how perturbation and gradient based methods relate to counterfactual explanations.** Furthermore, we mainly focused on the **analysis of feature attribution methods.** It would be exciting to analyze other kinds of explanation methods such as rule based or prototype based methods.