This is the professional `README.md` file your project deserves.

It explains **what** the project is, **why** it matters, and gives the exact commands your group members need to run it.

**Action:** Go to your GitHub repository, click the pencil icon ✏️ on your `README.md` file, delete everything, and paste this in.

---

```markdown
# 🏭 TE Connectivity: Digital Twin Scrap Prediction System

### A Dual-Mode AI for Manufacturing Optimization
**Live Dashboard:** Built with Python (Streamlit) & XGBoost

---

## 📖 Project Overview
This project is an **Industrial Digital Twin** designed to predict manufacturing scrap in injection molding machines before it happens. It solves the problem of reactive quality control by using AI to forecast defects based on historical data and real-time sensor physics.

### 🧠 The Core Architecture
The system operates in two distinct modes:

1.  **📅 Planning Mode (The Manager's View):**
    * **Goal:** Strategic budgeting.
    * **How it works:** Uses 6 months of historical production data to predict the expected scrap rate for a specific Machine, Tool, and Part combination.
    * **Value:** Helps managers schedule maintenance for high-risk tools before a shift begins.

2.  **🔴 Monitoring Mode (The Operator's View):**
    * **Goal:** Real-time intervention.
    * **How it works:** Connects to a live simulation of machine sensor logs (Temperature, Cushion, Pressure). An XGBoost model analyzes these "vital signs" second-by-second.
    * **Value:** If the physics drift (e.g., Cushion instability), the system triggers a **"HIGH RISK"** alert instantly, allowing the operator to stop the machine and save thousands of parts.

---

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Real-time Dashboard)
* **Machine Learning:** XGBoost Regressor (Gradient Boosting)
* **Data Processing:** Pandas (ETL Pipeline for Sensor Logs)
* **Visualization:** Plotly (Interactive Charts)

---

## 🚀 How to Run Locally (Step-by-Step)

Follow these instructions to set up the project on your own machine.

### Prerequisites
* Python 3.8+ installed (Make sure to check "Add Python to PATH" during installation).
* VS Code (Recommended IDE).
* Git.

### 1. Clone the Repository
Open your terminal (or Git Bash) and run:
```bash
git clone [https://github.com/Atharva-cell-web/scrap-prediction-tool-main.git](https://github.com/Atharva-cell-web/scrap-prediction-tool-main.git)
cd scrap-prediction-tool-main

```

### 2. Set Up the Virtual Environment

Create an isolated Python environment to keep your system clean.

**For Windows:**

```powershell
python -m venv env
.\env\Scripts\activate

```

**For Mac/Linux:**

```bash
python3 -m venv env
source env/bin/activate

```

*(You should see `(env)` appear at the start of your terminal line).*

### 3. Install Dependencies

Install all required libraries automatically:

```bash
pip install -r requirements.txt

```

### 4. Run the Dashboard

Launch the application:

```bash
streamlit run forecasting_v2/app_v3.py

```

The app will automatically open in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

* `forecasting_v2/data/` - Contains the raw sensor logs and historical scrap reports.
* `forecasting_v2/models/` - Stores the trained AI brains (`.pkl` files).
* `forecasting_v2/scripts/` - The Python code used to train the models.
* `app_v3.py` - The main dashboard application.

---

## 👨‍💻 Troubleshooting

* **Error: "Streamlit is not recognized"**
* *Fix:* Make sure you activated the environment (Step 2) before running the command.


* **Error: "Path not found"**
* *Fix:* Ensure you are in the root folder (`scrap-prediction-tool-main`) before running the streamlit command.



---

### © 2026 TE Digital Twin Team

```

```
