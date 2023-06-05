
# XAI via L2O

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Docs](https://github.com/TypalAcademy/xai-l2o/actions/workflows/ci.yml/badge.svg)

:material-draw-pen: Howard Heaton and Samy Wu Fung

!!! note "Summary"
    We fuse optimization-based deep learning models to give explainability with output guarantees and certificates of trustworthiness.

!!! success "Key Steps"
    - [x] Create an optimization model with data-driven and analytic terms: $\mathsf{\mathcal{N}_\Theta(d) = \underset{x}{\text{argmin}}\  f_\Theta(x;\ d)}$
    - [x] Identify an optimization algorithm for the model: $\mathsf{x^{k+1} = T_\Theta(x^k;\ d)}$
    - [x] Train using JFB

<center>
[Preprint :fontawesome-solid-file-lines:](assets/xia-via-l2o-preprint.pdf){ .md-button .md-button--primary }
[Slides :fontawesome-solid-file-image:](assets/xai-via-l2o-slides.pdf){ .md-button .md-button--primary }
</center>

!!! abstract "Abstract"

    Indecipherable black boxes are common in machine learning (ML), but applications increasingly require explainable artificial intelligence (XAI). The core of XAI is to establish transparent and interpretable data-driven algorithms. This work provides concrete tools for XAI in situations where prior knowledge must be encoded and untrustworthy inferences flagged. We use the "learn to optimize" (L2O) methodology wherein each inference solves a data-driven optimization problem. Our L2O models are straightforward to implement, directly encode prior knowledge, and yield theoretical guarantees (e.g. satisfaction of constraints). We also propose use of interpretable certificates to verify whether model inferences are trustworthy. Numerical examples are provided in the applications of dictionary-based signal recovery, CT imaging, and arbitrage trading of cryptoassets.

!!! quote "Citation"
    ```
    @article{heaton2022explainable,
             title={{Explainable AI via Learning to Optimize}},
             author={Heaton, Howard and Wu Fung, Samy},
             journal={arXiv preprint arXiv:2204.14174},
             year={2022}
    }
    ```
