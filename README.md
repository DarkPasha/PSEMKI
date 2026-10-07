# PSEMKI — Cats vs. Dogs Classifier

A convolutional neural network that tells cats from dogs, written in PyTorch for a school **P-Seminar on artificial intelligence** (2022). The project includes the training script with a live error-rate plot and a small Tkinter app where you pick a photo and the network says *"Dein Tier ist: Hund"* or *"… Katze"*.

## How it works

- **Data:** the Kaggle/Microsoft *PetImages* dataset (cats and dogs), resized to 256 px and center-cropped, normalised with the ImageNet mean and standard deviation.
- **Model:** a 5-layer CNN (`Conv2d` 3→6→12→18→24→56 with max-pooling) followed by two fully connected layers (1400 → 1000 → 2).
- **Training:** the number of epochs is entered at start-up; the error rate per epoch is plotted with matplotlib. There are two variants: one for a PC with an NVIDIA GPU (CUDA) and one that runs on a laptop CPU.

## Repository layout

| Path | Contents |
| --- | --- |
| `Abgabe_20/PC_KI_lernprozess/` | Training script, GPU (CUDA) version, plus an error-dialog helper |
| `Abgabe_20/Laptop_KI_Lernprozess/` | Same training script for CPU |
| `Abgabe_20/KI_testen/` | Tkinter app that loads the trained model (`NetzTest.pt`) and classifies a chosen image |
| `alte_versionen/` | Earlier experiments: first networks, a road-sign test, a small web front-end, presentation notes |

## Running it

```bash
pip install torch torchvision pillow matplotlib

# train (expects PetImages/training_data/ and PetImages/test_data/ next to the script)
python Abgabe_20/Laptop_KI_Lernprozess/cadwithgui.py

# classify a photo with the trained model (run from the repository root)
python Abgabe_20/KI_testen/app.py
```

> The model path in `KI_testen/app.py` uses Windows separators (`Abgabe_20\KI_testen\NetzTest.pt`); on macOS/Linux change it to forward slashes.

## Team

Built by a team of students for the P-Seminar KI, 2022. Training code by Emirhan and Daniel (laptop version with Kartik), classifier app by Kartik.
