# Evaluate test images stored in class folders

from tensorflow.keras.models import load_model  # Load a saved Keras model.
import cv2  # Image operations.
import numpy as np  # Array operations.
import os  # File and folder operations.

# Suppress scientific notation in NumPy output.
np.set_printoptions(suppress=True)

# Select the model folder; use the same test set for model comparisons.
letter = "b"
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = "model_" + letter

# Load the model for inference; the filename is case-sensitive.
model = load_model(
    base_dir + "/" + model_path + "/keras_model_" + letter + ".h5", compile=False
)

# Load labels in the same order as the model outputs.
class_names = open(
    base_dir + "/" + model_path + "/labels_" + letter + ".txt", "r"
).readlines()

# Remove each label's single-digit index and surrounding whitespace.
for idx in range(len(class_names)):
    class_names[idx] = class_names[idx][2:].strip()

# Class names must match the test folder names and model labels.
test_dir = base_dir
classes = ["rose", "tulip"]

# Track the total and correct predictions for each true class.
total_per_class = {c: 0 for c in class_names}
correct_per_class = {c: 0 for c in class_names}

# Store correct and incorrect predictions in separate folders.
result_dir = base_dir + "/results/" + model_path
correct_dir = os.path.join(result_dir, "correct")
wrong_dir = os.path.join(result_dir, "wrong")
os.makedirs(correct_dir, exist_ok=True)
os.makedirs(wrong_dir, exist_ok=True)

# Use each test folder's name as the ground-truth label.
for true_label in classes:
    folder_path = os.path.join(test_dir, true_label + "_test")

    for filename in os.listdir(folder_path):
        # Skip files with unsupported extensions.
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(folder_path, filename)

        # Read the original BGR image; a valid image file is assumed.
        image_ = cv2.imread(img_path)

        # Resize to the model input size.
        image = cv2.resize(image_, (224, 224), interpolation=cv2.INTER_AREA)

        # Convert to float32 and add a batch dimension: (1, 224, 224, 3).
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Scale pixels to [-1, 1]. The BGR order must match training.
        image = (image / 127.5) - 1

        # Predict class scores and select the highest-scoring class.
        prediction = model.predict(image)
        pred_index = np.argmax(prediction)
        pred_label = class_names[pred_index]
        confidence_score = prediction[0][pred_index]

        # Add the predicted class and confidence to the original image.
        image_ = cv2.putText(
            image_, pred_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        image_ = cv2.putText(
            image_,
            str(np.round(confidence_score * 100))[:-2] + "%",
            (200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        # Select the output folder based on prediction correctness.
        if pred_label == true_label:
            save_folder = correct_dir
        else:
            save_folder = wrong_dir

        # Keep the original filename; duplicate names may be overwritten.
        save_name = filename
        save_path = os.path.join(save_folder, save_name)
        cv2.imwrite(save_path, image_)

        # Count all test images and correct predictions by true class.
        total_per_class[true_label] += 1
        if pred_label == true_label:
            correct_per_class[true_label] += 1

        # GT: ground truth; Pred: predicted class.
        print(f"[GT: {true_label:10s}]  Pred: {pred_label:10s}")

# Report per-class accuracy; every class must contain test images.
print("\n===== Per-Class Accuracy =====")
for cls in class_names:
    acc = correct_per_class[cls] / total_per_class[cls] * 100
    print(
        f"{cls:10s}: {acc:6.2f}% " f"({correct_per_class[cls]}/{total_per_class[cls]})"
    )

# Combine counts across all classes.
total_correct = sum(correct_per_class.values())
total_images = sum(total_per_class.values())

# Overall accuracy = correct predictions / total images * 100.
overall_acc = total_correct / total_images * 100

print("\n===== Overall Accuracy =====")
print(f"Overall: {overall_acc:.2f}% ({total_correct}/{total_images})")
