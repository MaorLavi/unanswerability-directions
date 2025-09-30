# Detecting (Un)answerability in Large Language Models with Linear Directions

This repository contains the official code of the paper: ["Detecting (Un)answerability in Large Language Models with Linear Directions"](https://arxiv.org/abs/2509.22449).

### Citation
@misc{lavi2025detectingunanswerabilitylargelanguage,
      title={Detecting (Un)answerability in Large Language Models with Linear Directions}, 
      author={Maor Juliet Lavi and Tova Milo and Mor Geva},
      year={2025},
      eprint={2509.22449},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.22449}, 
}

## Setup
Our experiments were conducted in a **Python 3.11** environment. To clone the repository and set up the environment, please run the following commands:
```bash
git clone https://github.com/MaorLavi/unanswerability-directions.git
cd unanswerability-directions
pip install -r requirements.txt
```

## Notebooks
Use the following notebooks to run the code and reproduce results:
- demo.ipynb — derive linear directions and run direction-based classification.
- prompt_baselines.ipynb — prompt-based baselines and evaluation.
- additional_experiments.ipynb — extended experiments and analysis.

## Data
The processed splits for SQuAD 2.0, NQ, MuSiQue, and RepLiQA are included under data/.