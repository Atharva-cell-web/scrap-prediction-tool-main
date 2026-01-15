


```markdown
# 🏭 Scrap Risk Prediction Tool | TE Connectivity AI Cup 2026

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange) ![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project is developed for the **TE Connectivity AI Cup 2026**. It utilizes Machine Learning (Random Forest & Gradient Boosting) to analyze sensor data from the manufacturing process and predict the risk of "scrap" (defective products) early in the production line.

**Key Features:**
* **Data Preprocessing:** Handles missing values and feature scaling.
* **Model Comparison:** Evaluates Logistic Regression, Random Forest, and Gradient Boosting.
* **High Accuracy:** Achieved **99.9% accuracy** on the validation set using Random Forest.
* **Automated Pipeline:** Scripts for training, validation, and result visualization.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-learn, Matplotlib, Seaborn
* **Tools:** VS Code, Git

---

## 🚀 How to Run Locally

Follow these steps to set up the project on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/Atharva-cell-web/scrap-prediction-tool-main.git](https://github.com/Atharva-cell-web/scrap-prediction-tool-main.git)
cd scrap-prediction-tool-main

```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt

```

### 3. Add the Dataset (Important)

*Note: Due to file size limits, the dataset is not included in this repo.*

1. Create a folder named `data` inside the project directory.
2. Place your dataset file (e.g., `scrap_data.csv`) inside the `data` folder.
3. Create an empty folder named `models` in the root directory (this is where trained models will be saved).

### 4. Run the Training Script

To train the model and see the evaluation metrics:

```bash
python scripts/train_final_model.py

```

To compare different algorithms:

```bash
python scripts/compare_models.py

```

---

## 📂 Project Structure

```
├── data/                  # Dataset files (Not on GitHub)
├── models/                # Trained .pkl models (Not on GitHub)
├── scripts/               # Python source code
│   ├── train_final_model.py
│   ├── compare_models.py
│   └── validate_model.py
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```

## 👨‍💻 Contributors

* **Atharva Patil** - *Lead Developer*



```
