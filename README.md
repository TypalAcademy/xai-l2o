[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Docs](https://github.com/howardheaton/xai/actions/workflows/ci.yml/badge.svg)

# XAI-L2O: Explainable AI via Learning to Optimize

Indecipherable black boxes are common in machine learning (ML), but applications increasingly require explainable artificial intelligence (XAI). The core of XAI is to establish transparent and interpretable data-driven algorithms. This work provides concrete tools for XAI in situations where prior knowledge must be encoded and untrustworthy inferences flagged. We use the "learn to optimize" (L2O) methodology wherein each inference solves a data-driven optimization problem. Our L2O models are straightforward to implement, directly encode prior knowledge, and yield theoretical guarantees (e.g. satisfaction of constraints). We also propose use of interpretable certificates to verify whether model inferences are trustworthy. Numerical examples are provided in the applications of dictionary-based signal recovery, CT imaging, and arbitrage trading of cryptoassets.

## Publication

_Explainable AI via Learning to Optimize_ (**[arXiv Link](https://arxiv.org/abs/2204.14174)**)

    @article{heaton2022explainable,
      title={{Explainable AI via Learning to Optimize}},
      author={Heaton, Howard and Wu Fung, Samy},
      journal={arXiv preprint arXiv:2204.14174},
      year={2022}
    }

See [documentation site](https://howardheaton.github.io/xai/) for more details.