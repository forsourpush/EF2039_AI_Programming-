# Rose and Tulip Classification with Teachable Machine

This project investigates how training dataset size and label errors affect rose and tulip classification.

## Data and Experiments

Dataset: [Flowers Recognition by Alexander Mamaev, Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

| Model | Training Images per Class | Label Error Rate |
|---|---:|---:|
| A | 30 | 0% |
| B | 100 | 0% |
| C | 250 | 0% |
| D | 250 | 10% |
| E | 250 | 20% |

A–C compare training data quantity. C–E compare label error rates using the same images. 25 images per class are assigned to the opposite class for D and 50 per class for E.

All models use the same test set with 100 rose and 100 tulip images. These images are excluded from every training set.

## Files in This Repository

- `model_a/`–`model_e/`: Trained models and labels, named `keras_model_a.h5`, `labels_a.txt`, and so on.
- `rose_a/`–`rose_e/`, `tulip_a/`–`tulip_e/`: Training images.
- `rose_test/`, `tulip_test/`: Test images.
- `evaluate_models.py`: Evaluation code.
- `requirements.txt`: Package versions.
- `results/`: Contains correct and incorrect predictions images.

## How to Download and Set Up

1. Clone the repository. Model weights are included:

   ```powershell
   git clone https://github.com/forsourpush/EF2039_AI_Programming-.git
   cd EF2039_AI_Programming-
   ```
   Download [image_data.zip](https://1drv.ms/u/c/097eb923a7555516/IQBxq7u3rZQAR4KnU2DldY3eAYuFUBOXti9LLwRIQQvbevo?e=6DnWtw) and extract its contents
into the repository root, together with `evaluate_models.py`.

2. Create and activate a Conda environment:

   ```powershell
   conda create -n flower_eval python=3.11.5 -y
   conda activate flower_eval
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

Evaluation was performed on **Windows with Python 3.11.5**. It runs on a CPU without a dedicated GPU. 

## How to Reproduce Training

Repeat these steps for each model A–E:

1. Open [Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Select **Image Project → Standard image model**.
3. Create two classes: `rose` and `tulip`.
4. Upload the corresponding training folders, keeping their assigned labels.
5. Set **epochs: 50**, **batch size: 16**, and **learning rate: 0.001**.
6. Train and select **Export Model → TensorFlow → Keras**.
7. Save the exported files in the corresponding model folder, using the filenames listed above.

Each model was trained once. Retraining may produce different results.

## How to Run Evaluation

1. Open `evaluate_models.py` and select the model:

   ```python
   letter = "a"
   ```

2. Run:

   ```powershell
   python evaluate_models.py
   ```

3. Repeat with `"b"`, `"c"`, `"d"`, and `"e"`.

The script reads files relative to its location. Keep the model and test folders together.

Evaluation follows the course script: **BGR images, 224 × 224 resizing, and normalization to [-1, 1]**. The terminal reports per-class recall under “Per-Class Accuracy” and overall accuracy. Prediction images are saved in `results/model_<letter>/correct/` and `results/model_<letter>/wrong/`.

## Expected Results

Using the supplied models and the fixed 200-image test set:

| Model | Rose Recall | Tulip Recall | Overall Accuracy |
|---|---:|---:|---:|
| A | 70% | 85% | 75.5% |
| B | 85% | 86% | 85.5% |
| C | 95% | 83% | 89% |
| D | 75% | 83% | 79% |
| E | 54% | 68% | 61% |