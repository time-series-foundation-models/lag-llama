# Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting

![lag-llama-architecture](images/lagllama.webp)

Lag-Llama is the <b>first open-source foundation model for time series forecasting</b>!

[[Tweet Thread](https://twitter.com/arjunashok37/status/1755261111233114165)] [[Model Weights](https://huggingface.co/time-series-foundation-models/Lag-Llama)] [[Colab Demo 1: Zero-Shot Forecasting](https://colab.research.google.com/drive/1XxrLW9VGPlZDw3efTvUi0hQimgJOwQG6?usp=sharing)] [[Colab Demo 2: (Preliminary Finetuning)](https://colab.research.google.com/drive/1uvTmh-pe1zO5TeaaRVDdoEWJ5dFDI-pA?usp=sharing)] [[GitHub](https://github.com/time-series-foundation-models/lag-llama)] [[Paper](https://arxiv.org/abs/2310.08278)]

____
This repository houses the Lag-Llama architecture.

____

* **Coming Next**: Detailed Fine-tuning Tutorial with examples on real-world datasets and best practices in using Lag-Llama! (Coming ~~by end of March~~ soon)

<b>Updates</b>:

* **5-Apr-2024**: We have updated our requirements file with new versions of certain packages. Please update/recreate your environments if you have previously used the code locally.
* **7-Mar-2024**: We have released a preliminary [Colab Demo 2](https://colab.research.google.com/drive/1uvTmh-pe1zO5TeaaRVDdoEWJ5dFDI-pA?usp=sharing) for finetuning, while we prepare a detailed tutorial. Please note this is preliminary and cannot be used for benchmarking. A detailed demo with instructions for benchmarking is coming soon along with the tutorial. 
* **17-Feb-2024**: We have released a new updated [Colab Demo 1](https://colab.research.google.com/drive/1XxrLW9VGPlZDw3efTvUi0hQimgJOwQG6?usp=sharing) for zero-shot forecasting that shows how one can load time series of different formats.
* **7-Feb-2024**: We released Lag-Llama, with open-source model checkpoints and a Colab Demo for zero-shot forecasting.

____

Current Features:

💫 <b>Zero-shot forecasting</b> on a dataset of <b>any frequency</b> for <b>any prediction length</b>, using <a href="https://colab.research.google.com/drive/1XxrLW9VGPlZDw3efTvUi0hQimgJOwQG6?usp=sharing" target="_blank">Colab Demo 1.</a><br/>

💫 (Preliminary) <b>Finetuning</b> on a dataset using [Colab Demo 2](https://colab.research.google.com/drive/1uvTmh-pe1zO5TeaaRVDdoEWJ5dFDI-pA?usp=sharing).

____

Coming Soon:

⭐ A tutorial for <b>finetuning</b> Lag-Llama.

⭐ A tutorial for <b>pretraining</b> Lag-Llama on your own large-scale data.

⭐ Scripts to <b>reproduce</b> all results in the paper.

⭐ An <b>online gradio demo</b> where you can upload time series and get zero-shot predictions and perform finetuning.

____

Stay Tuned!🦙

____

## Contact

We are dedicated to ensuring the reproducility of our results, and would be happy to help clarify questions about benchmarking our model or about the experiments in the paper.
The quickest way to reach us would be by email. Please email **both**: 
1. [Arjun Ashok](https://ashok-arjun.github.io/) - arjun [dot] ashok [at] servicenow [dot] com
2. [Kashif Rasul](https://scholar.google.de/citations?user=cfIrwmAAAAAJ&hl=en) - kashif [dot] rasul [at] gmail [dot] com

If you have questions about the model usage (or) code (or) have specific errors (eg. using it with your own dataset), it would be best to create an issue in the GitHub repository.

## Citing this work

Please use the following Bibtex entry to cite Lag-Llama.

```
@misc{rasul2024lagllama,
      title={Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting}, 
      author={Kashif Rasul and Arjun Ashok and Andrew Robert Williams and Hena Ghonia and Rishika Bhagwatkar and Arian Khorasani and Mohammad Javad Darvishi Bayazi and George Adamopoulos and Roland Riachi and Nadhir Hassen and Marin Biloš and Sahil Garg and Anderson Schneider and Nicolas Chapados and Alexandre Drouin and Valentina Zantedeschi and Yuriy Nevmyvaka and Irina Rish},
      year={2024},
      eprint={2310.08278},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```




