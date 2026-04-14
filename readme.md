# 🧠 PCB Defect Evaluation with LLM Explanations

## 📌 Overview

This project evaluates **PCB (Printed Circuit Board) defect classification** using a multimodal Large Language Model (**LLaVA 1.5-7B**).

Unlike traditional pipelines, this system not only predicts defects but also **explains its own mistakes** when it misclassifies an image.

> 💡 Key idea:
> After every wrong prediction, the same model is queried again to generate a **human-like explanation** of why the error occurred.

---

## 🎯 Objectives

* Perform **zero-shot and few-shot classification** of PCB defects
* Compare performance between prompting strategies
* Generate **LLM-based explanations** for incorrect predictions
* Visualize results using:

  * Confusion matrices
  * Annotated defect images
  * Prediction grids

---

## 🧩 Dataset

The project uses a PCB defect dataset with:

* 📷 Images of PCB boards
* 📦 XML annotations (bounding boxes)
* 🏷️ 6 defect classes:

| Class           | Description                  |
| --------------- | ---------------------------- |
| Mouse_bite      | Missing notch in copper edge |
| Open_circuit    | Broken trace                 |
| Short           | Unwanted connection          |
| Spur            | Thin copper spike            |
| Missing_hole    | Missing drill hole           |
| Spurious_copper | Extra copper                 |

---

## ⚙️ Pipeline

### 1️⃣ Data Processing

* Parse XML annotations
* Extract bounding boxes
* Crop defect regions
* Overlay:

  * 🔴 Red bounding box
  * 🟡 Corner markers
  * 🏷️ Label banner

---

### 2️⃣ Model

* Model: **LLaVA 1.5-7B**
* Loaded in **4-bit quantization (bitsandbytes)** for efficiency

---

### 3️⃣ Classification

Two strategies:

#### 🔹 Zero-Shot

* Uses detailed prompt with defect descriptions

#### 🔹 Few-Shot

* Uses example-based prompting

---

### 4️⃣ Self-Explanation (Core Innovation)

If prediction is wrong:

➡️ Model is re-queried with:

* True label
* Predicted label
* Same image

➡️ Output:

* Explanation of confusion
* Visual reasoning (edges, shapes, patterns)

---

### 5️⃣ Evaluation Metrics

* Accuracy
* Precision / Recall / F1-score
* Confusion Matrix
* Unknown prediction rate

---

### 6️⃣ Visualization

* 📊 Confusion matrices
* 🖼️ Annotated crops
* 🧾 Prediction grid (correct vs incorrect)

---

## 🚀 How to Run

### 🔧 Requirements

```bash
pip install torch transformers bitsandbytes seaborn scikit-learn pillow matplotlib
```

### ▶️ Run

```bash
python pcb_eval_with_explanations.py
```

---

## 📂 Output Files

Saved in:

```
/kaggle/working/
```

Includes:

* ✅ Annotated defect images
* 📊 Confusion matrices
* 🖼️ Prediction grids

---

## 💡 Key Insights

* Few-shot prompting generally improves accuracy
* Model often confuses:

  * Open circuit vs Short
  * Spur vs Spurious copper
* Explanations reveal **visual ambiguity patterns**

---

## ⚠️ Limitations

* Depends heavily on prompt quality
* Explanations may not always be fully reliable
* Computationally expensive (LLM inference)

---

## 🔮 Future Work

* Fine-tuning on PCB dataset
* Better prompt engineering
* Automated error analysis dashboard
* Integration with real-time inspection systems

---

