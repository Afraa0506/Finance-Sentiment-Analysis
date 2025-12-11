# 📈 Finance Sentiment Analysis App

A complete NLP project that predicts the sentiment of **financial text** using a **TF-IDF Vectorizer** and a **Linear SVC** model.
This project includes:

* 🧠 **ML Model** (TF-IDF + Linear SVC)
* ⚡ **FastAPI Backend** for prediction API
* 🎨 **Streamlit Web UI** with custom styling
* 📁 Clean folder structure for deployment

---

## 🚀 Features

✔ Classifies financial statements into **Positive**, **Neutral**, or **Negative**
✔ Uses **TF-IDF** features
✔ Model trained using **Linear SVC**
✔ FastAPI endpoint for external apps
✔ Beautiful Streamlit interface
✔ Real-time predictions
✔ Finance-friendly, colorful UI

---


## 🧠 Model Details

* **Vectorizer:** TF-IDF (unigrams + bigrams)
* **Classifier:** Linear Support Vector Classifier (Linear SVC)
* **Classes:**

  * `0` → Negative
  * `1` → Neutral
  * `2` → Positive

---

## 🏗️ How to Install

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the FastAPI Backend

```bash
uvicorn main:app --reload
```

API will run at:

```
http://127.0.0.1:8000/predict
```

### Example Request (JSON)

```json
{
  "text": "The company's earnings improved significantly this quarter."
}
```

---

## 🌐 Run the Streamlit App

```bash
streamlit run app.py
```

This opens the UI in your browser.

---

## 🖼️ UI Preview

* Custom background
* Pastel finance-themed colors
* Banner image
* Styled buttons and text

---

## 📊 Dataset Used

**Financial PhraseBank** — a popular dataset for financial sentiment analysis containing sentences labeled as Positive, Neutral, or Negative.

---

## 📌 Usage

Ideal for:

* Stock market reports
* Financial news sentiment
* Business document classification
* Market analysis dashboards

---

## Screenshots

<img width="600" height="783" alt="Screenshot 2025-12-11 185529" src="https://github.com/user-attachments/assets/6e779d87-5a44-42bd-9b1f-0cc279c739c7" />
<img width="595" height="772" alt="Screenshot 2025-12-11 185636" src="https://github.com/user-attachments/assets/83bd9db0-788f-4771-9a06-02e7a8863663" />
<img width="599" height="771" alt="Screenshot 2025-12-11 185710" src="https://github.com/user-attachments/assets/8a74dad0-f2a7-4f3b-b050-fe01a3217317" />




---
