# Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting

![lag-llama-architecture](images/lagllama.webp)

Lag-Llama is the <b>first open-source foundation model for time series forecasting</b>!

[[Tweet Thread](https://twitter.com/arjunashok37/status/1755261111233114165)] [[Model Weights](https://huggingface.co/time-series-foundation-models/Lag-Llama)] [[Colab Demo on Zero-Shot Forecasting](https://colab.research.google.com/drive/1XxrLW9VGPlZDw3efTvUi0hQimgJOwQG6?usp=sharing)] [[GitHub](https://github.com/time-series-foundation-models/lag-llama)] [[Paper](https://arxiv.org/abs/2310.08278)]

____
This repository houses the Lag-Llama architecture.

____

* **Coming Next**: Fine-tuning scripts with examples on real-world datasets and best practices in using Lag-Llama!🚀  

<b>Updates</b>:

* **17-Feb-2024**: We have released a new updated [Colab Demo](https://colab.research.google.com/drive/1XxrLW9VGPlZDw3efTvUi0hQimgJOwQG6?usp=sharing) for zero-shot forecasting that shows how one can load time series of different formats.
* **7-Feb-2024**: We released Lag-Llama, with open-source model checkpoints and a Colab Demo for zero-shot forecasting.

____

Current Features:

💫 <b>Zero-shot forecasting</b> on a dataset of <b>any frequency</b> for <b>any prediction length</b>, using the <a href="https://colab.research.google.com/drive/1XxrLW9VGPlZDw3efTvUi0hQimgJOwQG6?usp=sharing" target="_blank">Colab Demo.</a><br/>


<b>Fine-tuning</b> on a dataset: [Colab Demo](https://colab.research.google.com/drive/1uvTmh-pe1zO5TeaaRVDdoEWJ5dFDI-pA?usp=sharing)

____

Coming Soon:

⭐ An <b>online gradio demo</b> where you can upload time series and get zero-shot predictions and perform finetuning.

⭐ Features for <b>finetuning</b> the foundation model

⭐ Features for <b>pretraining</b> Lag-Llama on your own large-scale data

⭐ Scripts to <b>reproduce</b> all results in the paper.


____

Stay Tuned!🦙

____

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




