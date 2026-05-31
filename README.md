# Blood Cell Classification from Microscopic Images Using Shallow CNN

## Project Overview

This project automatically detects and classifies blood cells from microscopic images using Deep Learning.

The system identifies:

- Red Blood Cells (RBC)
- White Blood Cells (WBC)
- Platelets

The model uses a Shallow CNN / Deep Learning approach for automated blood cell analysis.

---

## Features

Upload Microscopic Blood Cell Images

Detect Blood Cells Automatically

Count RBC Cells

Count WBC Cells

Count Platelets

Reject Invalid Images

Display Detection Results Visually

---

## Dataset Used

BCCD Dataset

Contains:

- RBC Images
- WBC Images
- Platelet Images

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- Streamlit
- NumPy
- Pillow

---

## Project Structure

```text
blood-cell-classifier/

│ app.py

│ train.py

│ convert_dataset.py

│ blood_cnn.pth

│ requirements.txt

│ README.md
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/blood-cell-classifier.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

---

## Usage

1. Upload microscopic blood image

2. Click Predict

3. System detects cells automatically

4. View:

- RBC Count
- WBC Count
- Platelet Count

---

## Model Workflow

Input Image

↓

Preprocessing

↓

Cell Detection

↓

CNN Classification

↓

Cell Counting

↓

Result Visualization

---

## Applications

- Medical Laboratories

- Blood Cell Analysis

- Disease Detection Support

- Educational Use

- Automated Diagnostics

---

## Research Publication

Title:

**Classification of Blood Cells from Microscopic Images Using Shallow CNN**


## Future Scope

- Anemia Detection

- Leukemia Detection

- Abnormal Cell Identification

- Hospital Integration

---

## License

Educational and Research Purpose
