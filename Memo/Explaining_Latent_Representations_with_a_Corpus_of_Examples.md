# Explaining Latent Representations with a Corpus of Examples

	使用例子解释
	
	https://arxiv.org/pdf/2110.15355.pdf

[TOC]

## Abstract

Modern machine learning models are complicated. Most of them rely on convoluted latent representations of their input to issue a prediction. To achieve greater transparency than a black-box that connects inputs to predictions, **it is necessary to gain a deeper understanding of these latent representations.** To that aim, we propose SimplEx: a user-centred method that provides example-based explanations with reference to a freely selected set of examples, called the corpus. SimplEx uses the corpus to improve the user’s understanding of the latent space with post-hoc explanations answering two questions: (1) Which corpus examples explain the prediction issued for a given test example? (2) What features of these corpus
examples are relevant for the model to relate them to the test example? SimplEx provides an answer by reconstructing the test latent representation as a mixture of corpus latent representations. Further, we propose a novel approach, the Integrated Jacobian, that allows SimplEx to make explicit the contribution of each corpus feature in the mixture. Through experiments on tasks ranging from mortality prediction to image classification, we demonstrate that these decompositions are robust and accurate. **With illustrative use cases in medicine, we show that SimplEx empowers the user by highlighting relevant patterns in the corpus that explain model representations. Moreover, we demonstrate how the freedom in choosing the corpus allows the user to have personalized explanations in terms of examples that are meaningful for them.**



## 1. Introduction and related work

How can we make a machine learning model convincing? If accuracy is undoubtedly necessary, it is rarely sufficient. As these models are used in critical areas such as medicine, finance and the criminal justice system, their black-box nature appears as a major issue [1, 2, 3]. **With the necessity to address this problem, the landscape of explainable artificial intelligence (XAI) developed [4, 5].**

**In this work, we rather focus on post-hoc explainability techniques. These methods aim at improving the interpretability of black-box models by complementing their predictions with various kinds of explanations. In this way, it is possible to understand the prediction of a model without sacrificing its prediction accuracy.**

Feature importance explanations are undoubtedly the most widespread type of post-hoc explanations. Popular feature importance methods include SHAP [7, 8, 9], LIME [10], Integrated Gradients [11], Contrastive Examples [12] and Masks [13, 14, 15]. **These methods complement the model prediction for an input example with a score attributed to each input feature. This score reflects the importance of each feature for the model to issue its prediction. Knowing which features are important for a model prediction certainly provides more information on the model than the prediction by itself. However, these methods do not provide a reason as to why the model pays attention to these particular features.**

**Another approach is to contextualize each model prediction with the help of relevant examples. In fact, recent works [16] have demonstrated that human subjects often find example-based explanations more insightful than feature importance explanations.** Complementing the model’s predictions with relevant examples previously seen by the model is commonly known as **Case-Based Reasoning (CBR) [17, 18, 19]. The implementations of CBR generally involve models that create a synthetic representation of the dataset, where examples with similar patterns are summarized by prototypes [20, 21, 22]. At inference time, these models relate new examples to one or several prototypes to issue a prediction. In this way, the patterns that are used by the model to issue a prediction are made explicit with the help of relevant prototypes. A limitation of this approach is the restricted model architecture. The aforementioned procedure requires to opt for a family of models that rely on prototypes to issue a prediction. This family of model might not always be the most suitable for the task at hand. This motivates the development of generic post-hoc methods that make few or no assumption on the model.**

The most common approach to provide example-based explanations for a wide variety of models mirrors feature importance methods. **The idea is to complement the model prediction by attributing a score to each training example.** This score reflects the importance of each training example for the model to issue its prediction. **This score will typically be computed by simulating the effect of removing each training instance from the training set on the learned model.**

**We cite Concept Activation Vectors that create a dictionary between human friendly concepts (such as the presence of stripes in an image) and their representation in terms of latent vectors [27]. Another interesting contribution is the Deep k-Nearest Neighbors model that contextualizes the prediction for an example with its Nearest Neighbours in the space of latent variables, the latent space [28]. An alternative exploration of the latent space is offered by the representer theorem that allows, under restrictive assumptions, to use latent vectors to decompose a model’s prediction in terms of its training examples [29].**

**In each case, SimplEx highlights the role played by each feature of each corpus example in the latent space decomposition.** SimplEx centralizes many functionalities that, to the best of our knowledge, constitute a leap forward from the previous state of the art. (1) **SimplEx gives the user freedom to choose the corpus of examples whom with the model’s predictions are decomposed.** Unlike previous methods such as the representer theorem, there is no need for this corpus of examples to be equal to the model’s training set. This is particularly interesting for two reasons: (a) **the training set of a model is not always accessible (b) the user might want explanations in terms of examples that make sense for them.** For instance, a doctor might want to understand the predictions of a risk model in terms of patients they know. (2) **The decompositions of SimplEx are valid, both in latent and output space. We show that, in both cases, the corpus mixtures discovered by SimplEx offer significantly more precision and robustness than previous methods such as Deep k-Nearest Neighbors and the representer theorem.** (3) **SimplEx details the role played by each feature in the corpus mixture. This is done by introducing Integrated Jacobians, a generalization of Integrated Gradients that makes the contribution of each corpus feature explicit in the latent space decomposition. This creates a bridge between two research directions that have mostly developed independently: feature importance and example-based explanations [19, 30].**



## 2. SimplEx

In this section, we formulate our method rigorously. **Our purpose is to explain the black-box prediction for an unseen test example with the help of a set of known examples that we call the corpus.**

### 2.1 Preliminaries

**Assumption 2.1** (Black-box Restriction). We restrict to black-boxes $\mathbf{f}: \mathcal{X} \rightarrow \mathcal{Y}$ that can be decomposed as $\mathbf{f} = \mathbf{l} \circ \mathbf{g}$, where $\mathbf{g}: \mathcal{X} \rightarrow \mathcal{H}$ maps an input $\mathbf{x} \in \mathcal{X}$ to a latent $\mathbf{h}  = \mathbf{g}(\mathbf{x}) \in \mathcal{H}$ and $\mathbf{l}: \mathcal{H} \rightarrow \mathcal{Y}$ linearly maps a latent vector $\mathbf{h} \in \mathcal{H}$ to an output $\mathbf{y} = \mathbf{l}(\mathbf{h}) = \mathbf{A} \mathbf{h} \in \mathcal{Y}$. In the following, we call $\mathcal{H} \subseteq \mathbb{R}^{d_H}$ the *latent space*. Typically, this space has higher dimension than the output space $d_H > d_Y$.

Remark 2.1. In the context of deep-learning, this assumption requires that the last hidden layer maps linearly to the output. **While it is often the case, it is crucial in the following since we will use the fact that linear combinations in latent space correspond to linear combinations in output space. Our purpose is to gain insights on the structure of the latent space.**

Remark 2.2. for a normalizing map $\phi$ (typically SoftMax): $\mathbf{f} = \phi \circ \mathbf{l} \circ \mathbf{g}$. In this case, we ignore the normalizing map $\phi$ and define the output to be $\mathbf{y}=(\mathbf{l} \circ \mathbf{g})(\mathbf{x})$.

### 2.2 A corpus of examples to explain a latent representation

This hints that the latent space $\mathcal{H}$ is endowed with the appropriate geometry to make corpus decompositions.

**Definition 2.1** (Corpus Hull) The corpus convex hull spanned by a corpus $\mathcal{C}$ with latent representation $\mathbf{g}(\mathcal{C}) = \{\mathbf{h}(\mathbf{x}^c) | \mathbf{x}^c \in \mathcal{C}\}$ is the convex set
$$
\mathcal{C H}(\mathcal{C})=\left\{\sum_{c=1}^{c} w^{c} \mathbf{h}^{c} \mid w^{c} \in[0,1] \forall c \in[C] \wedge \sum_{c=1}^{c} w^{c}=1\right\}
$$
Remark 2.3. This is the set of latent vectors that are a mixture of the corpus latent vectors.

**Definition 2.2** (Corpus Residual) The corpus residual associated to a latent vector $\mathbf{h} \in \mathcal{H}$ and its corpus representation $\hat{\mathbf{h}} \in \mathcal{C} \mathcal{H} (\mathcal{C})$ solving (1) is the quantity
$$
r_{\mathcal{C}}(\mathbf{h})=\|\mathbf{h}-\hat{\mathbf{h}}\|_{\mathcal{H}}=\min _{\tilde{\mathbf{h}} \in \mathcal{C H}(\mathcal{C})}\|\mathbf{h}-\tilde{\mathbf{h}}\|_{\mathcal{H}}
$$




### 2.3 Transferring the corpus explanation in input space

Now that we are endowed with a corpus decomposition $\hat{\mathbf{h}} = \sum_{c=1}^C w^c \mathbf{h}^c$ that approximates $\mathbf{h}$, it would be convenient to have an understanding of the corpus decomposition in input space $\mathcal{X}$. For the sake of notation, we will assume that the corpus approximation is good so that it is unnecessary
to draw a distinction between the latent representation $\mathbf{h}$ of the unseen example $\mathbf{x}$ and its corpus decomposition $\hat{\mathbf{h}}$. If we want to understand the corpus decomposition in input space, a natural approach [11] is to fix a baseline input x0 together with its latent representation $\mathbf{h}^0 = \mathbf{g}(\mathbf{x}^0)$. Let us now decompose the representation shift $\mathbf{h} - \mathbf{h}^0$ in terms of the corpus: 
$$
\mathbf{h} - \mathbf{h}^0 = \sum_{c=1}^C w^c(\mathbf{h}^c - \mathbf{h}^0)
$$
To decompose the shift in latent space in terms of the features, we parametrize the shift in input space with a line $\gamma^c : [0; 1] \rightarrow \mathcal{X}$ that goes from the baseline to the corpus example: $\gamma^c(t) = \mathbf{x}^0 + t \cdot (\mathbf{x}^c - \mathbf{x}^0)$ for $t \in [0, 1]$. Together with the black-box, this line induces a curve in latent space $\mathbf{g} \circ \gamma^{c}:[0,1] \rightarrow \mathcal{H}$ that goes from the baseline latent representation $\mathbf{h}^0$ to the corpus example latent representation $\mathbf{h}^c$. Let us now use an infinitesimal decomposition of this curve to make the contribution of each input feature explicit. If we assume that $\mathbf{g}$ is differentiable at $\gamma^c(t)$, we can use a first order approximation of the curve at the vicinity of $t \in (0, 1)$ to decompose the infinitesimal shift in latent space:
$$
\begin{aligned}
\underbrace{\mathbf{g} \circ \gamma^{c}(t+\delta t)-\mathbf{g} \circ \gamma^{c}(t)}_{\text {Infinitesimal shift in latent space }} &=\left.\left.\sum_{i=1}^{d_{X}} \frac{\partial \mathbf{g}}{\partial x_{i}}\right|_{\boldsymbol{\gamma}^{c}(t)} \frac{d \gamma_{i}^{c}}{d t}\right|_{t} \delta t+o(\delta t) \\
&=\left.\sum_{i=1}^{d_{X}} \frac{\partial \mathbf{g}}{\partial x_{i}}\right|_{\boldsymbol{\gamma}^{c}(t)}\left(x_{i}^{c}-x_{i}^{0}\right) \cdot \delta t+o(\delta t),
\end{aligned}
$$
where we used $\gamma_i^c(t) = x_i^0 + t \cdot (x_i^c - x_i^0)$ to obtain the second equality. In this decomposition, each input feature contributes additively to the infinitesimal shift in latent space. It follows trivially that the contribution of the input feature corresponding to input dimension $i \in [d_X]$ is given by 
$$
\delta \mathbf{j}_{i}^{c}(t)=\left.\left(x_{i}^{c}-x_{i}^{0}\right) \cdot \frac{\partial \mathbf{g}}{\partial x_{i}}\right|_{\boldsymbol{\gamma}^{c}(t)} \delta t \quad \in \mathcal{H}
$$
In order to compute the overall contribution of feature $i$ to the shift, we let $\delta t \rightarrow 0$ and we sum the infinitesimal contributions along the line 
$c$. If we assume that $\mathbf{g}$ is almost everywhere differentiable,this sum converges to an integral in the limit $\delta t \rightarrow 0$ . This motivates the following definitions.

**Definition 2.3** (Integrated Jacobian & Projection). The integrated Jacobian between a baseline $(\mathbf{x}^0, \mathbf{h}^0 = \mathbf{g}(\mathbf{x}^0))$ and a corpus example $(\mathbf{x}^c, \mathbf{h}^c = \mathbf{g}(\mathbf{x}^c)) \in \mathcal{X} \times \mathcal{H}$ associated to feature $i \in [d_X]$ is 
$$
\mathbf{j}_{i}^{c}=\left.\left(x_{i}^{c}-x_{i}^{0}\right) \int_{0}^{1} \frac{\partial \mathbf{g}}{\partial x_{i}}\right|_{\boldsymbol{\gamma}^{c}(t)} d t \quad \in \mathcal{H}
$$
where $\gamma^c(t) \equiv \mathbf{x}^0 + t \cdot (\mathbf{x}^c - \mathbf{x}^0)$ for $t \in [0, 1]$. This vector indicates the shift in latent space induced by feature $i$ of corpus example $c$ when comparing the corpus example with the baseline. To summarize this contribution to the shift $\mathbf{h} - \mathbf{h}^0$ described in (2), we define the *projected Jacobian*
$$
p_{i}^{c}=\operatorname{proj}_{\mathbf{h}-\mathbf{h}^{0}}\left(\mathbf{j}_{i}^{c}\right) \equiv \frac{\left\langle\mathbf{h}-\mathbf{h}^{0}, \mathbf{j}_{i}^{c}\right\rangle}{\left\langle\mathbf{h}-\mathbf{h}^{0}, \mathbf{h}-\mathbf{h}^{0}\right\rangle} \in \mathbb{R}
$$
where $<\cdot, \cdot>$ is an inner product for $\mathcal{H}$ and the normalization is chosen for the purpose of Proposition 2.1.

**Proposition 2.1** (Properties of Integrated Jacobians). Consider a baseline $(\mathbf{x}^0, \mathbf{h}^0 = \mathbf{g}(\mathbf{x}^0))$ and a test example together with their latent representation $(\mathbf{x}, \mathbf{h} = \mathbf{g}(\mathbf{x})) \in \mathcal{X} \times \mathcal{H}$. If the shift $\mathbf{h} - \mathbf{h}^0$ admits a decomposition (2), the following properties hold.
$$
(A): \sum_{c=1}^{C} \sum_{i=1}^{d_{X}} w^{c} \boldsymbol{j}_{i}^{c}=\boldsymbol{h}-\boldsymbol{h}^{0} \quad(B): \sum_{c=1}^{C} \sum_{i=1}^{d_{X}} w^{c} p_{i}^{c}=1 .
$$
These properties show that the integrated Jacobians and their projections are the quantities that we are looking for: they transfer the corpus explanation into input space. The first equality decomposes the shift in latent space in terms of contributions $w^c \mathbf{j}^c_i$ arising from each feature of each corpus example. The second equality sets a natural scale to the contribution of each feature. For this reason, it is natural to use $w^c p^c_i$ to measure the contribution of feature $i$ of corpus example $c$.



## 3. Experiments



## 4. Discussion

**We have introduced SimplEx, a method that decomposes the model representations at inference time in terms of a corpus.** Through several experiments, we have demonstrated that **these decompositions are accurate and can easily be personalized to the user. Finally, by introducing Integrated Jacobians, we have brought these explanations to the feature level.**

We believe that our bridge between feature and example-based explainability opens up many avenues for the future. **A first interesting extension would be to investigate how SimplEx can be used to understand latent representations involved in unsupervised learning.** For instance, **SimplEx could be used to study the interpretability of self-expressive latent representations learned by autoencoders [35]. A second interesting possibility would be to design a rigorous scheme to select the optimal corpus for a given model and dataset. Finally, a formulation where we allow the corpus to vary on the basis of observations would be particularly interesting for online learning.**