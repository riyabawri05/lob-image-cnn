# Seeing the Signal: Limit Order Book Price Prediction as an Image Problem

Can a plain CNN predict short-term price moves from a limit order book (LOB) if you simply turn the order book into a picture? On the FI-2010 benchmark, yes: our image-based CNN-LSTM reaches a weighted F1 of **87.93**, against **80.35** reported for DeepLOB under the same protocol.

This was a five-person group project for ST311 (Artificial Intelligence) at the London School of Economics, written up in NeurIPS format. Read the full [paper](seeing-the-signal-paper.pdf) or the [slides](seeing-the-signal-slides.pdf).

![A single LOB window encoded as a four-channel image](figures/sample_up.png)

## The idea

DeepLOB, the strongest published model on FI-2010, uses hand-designed (1×2) convolutions to pair each price with its volume and each bid with its ask before mixing across price levels. We asked whether you could skip that engineering by restructuring the data instead.

Each 100-step window of the order book (100 timesteps × 40 features) is rearranged into a **four-channel image of shape (4, 10, 100)**: bid volume, bid price, ask volume and ask price as separate channels, with the ten price levels stacked vertically and time running horizontally. Ask volumes are negated so the two sides of the book differ in sign, and prices are centred on the window's opening mid-price. Patterns such as a bid-ask imbalance then show up as local regions in the image, which is exactly what a convolution is good at picking up.

Two models share the same convolutional encoder (3×3 kernels, stride 1, no downsampling, then a tall (10×1) kernel that learns a weighting across all ten levels):

- **CNN-MLP** flattens the encoder output into a fully connected head, with no explicit temporal modelling.
- **CNN-LSTM** feeds a 64-dimensional feature vector per timestep into an LSTM.

LDA and a linear SVM on the final 40-feature snapshot serve as non-neural baselines.

## Results

FI-2010, Setup 2 (train on days 1 to 7, test on days 8 to 10), prediction horizon k = 50, pooled test set.

| Model | Accuracy | Precision | Recall | Weighted F1 |
|---|---|---|---|---|
| LDA (ours) | 41.63 | 44.95 | 41.63 | 42.25 |
| Linear SVM (ours) | 41.11 | 44.80 | 41.11 | 41.70 |
| C(TABL), Tran et al. 2019 | 74.81 | 74.58 | 74.27 | 74.32 |
| DeepLOB, Zhang et al. 2019 | 80.51 | 80.38 | 80.51 | 80.35 |
| **CNN-MLP (ours)** | 85.75 | 85.79 | 85.75 | **85.70** |
| **CNN-LSTM (ours)** | 88.02 | 88.36 | 88.02 | **87.93** |

Most of the gain over DeepLOB arrives before any recurrent layer is added, since the CNN-MLP alone beats it by 5.35 F1 points and the LSTM adds a further 2.23. That suggests the predictive structure in a 100-step window is largely spatial. The linear baselines, which only see the last snapshot, sit around 42, so the signal lives in how the book evolves across the window rather than in any single moment. Scores stayed within 3 F1 points across the three test days.

![CNN-LSTM training curves](figures/train_cnn_lstm.png)

### Caveats

FI-2010 covers five fairly illiquid Nordic stocks over two weeks in 2010, and models tuned on it are known to lose performance on more recent, more liquid markets. These results are evidence on this benchmark rather than a general claim. We also evaluated a single horizon (k = 50) and report single training runs, so variance across seeds isn't measured.

## Repository contents

```
lob_image_cnn.ipynb            full pipeline: data download, image encoding, baselines,
                               models, two-stage hyperparameter search, training, evaluation
seeing-the-signal-paper.pdf    NeurIPS-format write-up
seeing-the-signal-slides.pdf   presentation slides
figures/                       sample LOB images per class and training curves
requirements.txt
```

The notebook also contains `base` variants of each architecture (with max pooling), which we compared against the specialised `sup` versions reported above.

## Running it

The notebook was built for Google Colab on a GPU. It downloads the decimal-normalised FI-2010 data from the [DeepLOB repository](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) on first run, so no data is stored here. On Colab it caches intermediate results to Google Drive; elsewhere it writes to `./outputs`.

```bash
pip install -r requirements.txt
jupyter notebook lob_image_cnn.ipynb
```

Hyperparameters were tuned in two grid-search stages (learning rate and batch size, then dropout and weight decay) with AdamW, early stopping on validation loss and a fixed seed of 42. Selected values are listed in Appendix D of the [paper](seeing-the-signal-paper.pdf).

## My contribution

I wrote the data description, the methodology equations, algorithm box and schematic diagrams, the results section and the abstract. On the code side, I fixed and re-ran the pipeline and rewrote the testing step so all models are evaluated on the pooled test set across days 8 to 10, which is the protocol DeepLOB reports and the source of the headline numbers above.

## References

- Zhang, Zohren and Roberts (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. *IEEE Transactions on Signal Processing*.
- Ntakaris et al. (2018). Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods. *Journal of Forecasting*.

The full bibliography is in the paper.
