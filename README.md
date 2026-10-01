# EchoTrack: Whale Call Detection & Localization

An end-to-end sound event detection pipeline. It takes raw hydrophone (underwater microphone) audio, finds whale calls, tells you **which type of call** it is, and tells you **when it starts and ends**, with a confidence score for each detection.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/echotrack-whale-call-detection/blob/main/notebooks/echotrack.ipynb)

<!-- Put one picture here: a spectrogram with the detected calls marked on it. Save it as results/detections.png -->
![Example detections](results/detections.png)

## What it does

- Reads long continuous audio recordings (187+ hours in total)
- Cuts the audio into short windows and turns each window into a **log-Mel spectrogram** (an image of the sound)
- Uses a fine-tuned **ResNet18** to classify what is in each window (multi-label, 7 call types)
- Joins the window predictions into events with **start time, end time, call type and confidence**

## The hard parts

**1. Calls are rare.** Whale calls are present in under 10% of the audio, across 7 call types. A model can look "good" by just saying "no call" every time. The preprocessing pipeline is built around this class imbalance.

**2. Messy research code.** I fixed cross-platform audio decoding and multiprocessing failures in a third-party research codebase, then moved training to a cloud GPU with checkpointing so a crash didn't lose progress.

## Results

| Experiment | F1 on an unseen test site |
|---|---|
| Baseline | 0.10 |
| Weighted loss (class-imbalance fix) | 0.01 |

On the final run, the pipeline produced **372 timestamped detections across 4 call types**, including **204 overlapping events**.

### What went wrong with the weighted loss (and why I'm keeping it here)

The weighted-loss version did much *worse* than the baseline. I traced it to the weights: they grew without any limit, reaching up to **28,404x** for the rarest class, so training was dominated by a handful of examples. A **bounded** weighting fix is designed but **not yet run**. It's listed under "Next steps".

## How to run

1. Click the **Open in Colab** badge above.
2. Set the runtime to **GPU** (Runtime → Change runtime type).
3. Add the data: `[DESCRIBE WHERE THE AUDIO COMES FROM AND HOW TO PUT IT IN THE RIGHT FOLDER]`
4. Run the cells from top to bottom.

To run it on your own machine instead:

```bash
git clone https://github.com/YOUR-USERNAME/echotrack-whale-call-detection.git
cd echotrack-whale-call-detection
pip install -r requirements.txt
jupyter notebook notebooks/echotrack.ipynb
```

## Project structure

```
echotrack-whale-call-detection/
├── notebooks/
│   └── echotrack.ipynb      # the full pipeline
├── results/                 # plots and sample output
├── requirements.txt
└── README.md
```

## Tech stack

Python, PyTorch, Torchaudio, ResNet18 (transfer learning), Google Colab

## Next steps

- Run the bounded class-weighting fix and compare it with the baseline
- Test on more unseen sites

## Credits

Built on top of `[NAME AND LINK OF THE THIRD-PARTY RESEARCH CODEBASE / DATASET]`. Audio data is not included in this repo.

## Author

Irwin: [LinkedIn link] | [GitHub link]
