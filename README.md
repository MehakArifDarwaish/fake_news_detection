
# 📰 Fake News Detection App

This project is a **Fake News Detection System** built using **Machine Learning algorithms** and deployed with **Streamlit** for interactive use.  
It compares the performance of **Logistic Regression, Gradient Boosting, and Naive Bayes** for classifying news as *real* or *fake*.  

---

## 🚀 Features
- Train and evaluate ML models
- Visualize **Training vs Validation Loss**
- Compare performance across multiple classifiers
- Generate **WordCloud** for text visualization
- User-friendly **Streamlit App** for prediction

---

## 📂 Project Structure
```
FakeNewsDetection-App/
│── app.py                  # Streamlit app
│── model_training.py       # Training script
│── best_model.pkl          # Saved ML model
│── tfidf_vectorizer.pkl    # Saved vectorizer
│── requirements.txt        # Dependencies
│── README.md               # Project description
│── data/                   # Dataset (if included)
│── images/                 # Wordclouds & screenshots
```

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/<your-username>/FakeNewsDetection-App.git
cd FakeNewsDetection-App
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

Start the Streamlit app with:
```bash
streamlit run app.py
```

Then open the link shown in terminal (usually `http://localhost:8501`) in your browser.

---

## 📊 Models Used
- **Logistic Regression**
- **Gradient Boosting**
- **Naive Bayes**

Training and validation loss are visualized to compare model performance.

---

## 📈 Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.92     | 0.91      | 0.93   | 0.92     |
| Gradient Boosting    | 0.95     | 0.94      | 0.96   | 0.95     |
| Naive Bayes          | 0.89     | 0.88      | 0.90   | 0.89     |

*(Replace these values with your actual results)*

---

## 📷 Screenshots

(Add your screenshots here – e.g., model comparison graph, wordclouds, app interface.)

---

## 🛠️ Tech Stack
- Python 🐍
- Scikit-learn
- NLTK
- WordCloud
- Matplotlib & Seaborn
- Streamlit

---

## 👩‍💻 Author
**Hafiza Mehak Arif**  
AI Student | ML & Data Science Enthusiast  

📌 GitHub: [<your-username>](https://github.com/<your-username>)  
📌 LinkedIn: [Add your LinkedIn link here]  
