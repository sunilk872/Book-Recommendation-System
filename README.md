

# 📚 Book Recommendation System 

 https://book-recommendation-url.streamlit.app/ (app-URL)

## 🚀 Overview  
This project builds a **machine learning-based book recommendation system** that suggests books based on user interests, reading history, and preferences. It leverages **collaborative filtering, content-based filtering, and popularity-based models** to enhance user experience and help readers discover books they'll love.  

---

## 🎯 Objective  
✔ Extract key features from the **Book-Crossing dataset**  
✔ Provide **personalized book recommendations** using machine learning  
✔ Improve user engagement through **data-driven recommendations**  

---

## 📂 Dataset Overview   
📌 Dataset Link: 📂https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

The project utilizes **three datasets**: 
1️⃣ **Books Dataset 📚** → Contains details like title, author, publication year, image with different sizes(small, medium, large) and ISBN  
2️⃣ **Users Dataset 👤** → Includes user IDs, locations, and age details  
3️⃣ **Ratings Dataset ⭐** → Consists of book ratings provided by users  

📌 **Dataset Stats (After Preprocessing)**  
✔ **Books**: 271,360 records  
✔ **Users**: 278,858 records  
✔ **Ratings**: 1,149,780 records  

---

## 🛠️ Data Preprocessing  
✔ **Books Dataset:**  
   - Removed null values & unnecessary image URLs  
   - Fixed incorrect year values & standardized missing publishers  
   - Kept years between **1900 and 2025**  

✔ **Users Dataset:**  
   - Filtered ages between **5 and 105** to remove outliers  
   - Dropped **location column** (not relevant to model building)  

✔ **Ratings Dataset:**  
   - Cleaned ISBN column  
   - **Balanced dataset:** Sampled **50,000 unrated** and **150,000 rated books** for better performance  

---

## 📊 Data Visualization  
🔹 **Ratings Distribution** → Shows balance between rated & unrated books  
🔹 **Top-Rated Books** → Identifies books with the highest engagement  
🔹 **User Rating Patterns** → Understands user behavior for better recommendations  

---

## 🔍 Model Building  

1️⃣ **Popularity-Based Recommendation**  
   - Suggests books based on **overall popularity** (most rated & highest rated books)  

2️⃣ **Collaborative Filtering (KNN & Cosine Similarity)**  
   - Identifies **similar users** and recommends books based on user interactions  

3️⃣ **Content-Based Filtering (TF-IDF Vectorization)**  
   - Suggests books based on **similar content attributes** (title, author, genre, etc.)  

---

## 📏 Model Evaluation  
✔ Evaluated recommendation accuracy using:  
   - **Precision & Recall**  
   - **Hit Rate & Coverage**  
   - **User Satisfaction Score**  

---

## 🚀 Deployment  

📂 **Project Structure**  
```
|── 📁 data               # Raw & processed datasets
│── 📁 notebooks          # Jupyter notebooks for EDA & modeling  
|── 📁 deployment         # Deployment files & pickled models  
│── 📄 README.md          # Project documentation  
│── 📄 requirements.txt   # Dependencies  
```

### **🌍 Deployment Strategy**
- **Web App Deployment using Streamlit**  
- **Pickle File (.pkl) for Model Saving & Loading**  

Run the Streamlit app:  
```bash
streamlit run app.py
```

---

## 🛠️ Tools & Technologies  
✔ **Python** (Pandas, NumPy, Scikit-learn)  
✔ **Machine Learning** (KNN, Cosine Similarity, TF-IDF)  
✔ **Data Visualization** (Matplotlib, Seaborn)  
✔ **Deployment** (Streamlit)  

---

## 📌 How to Run  
1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/Book-Recommendation-System.git
cd Book-Recommendation-System
```
2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```
3️⃣ Run the model & deployment  
```bash
streamlit run app.py
```

---

## 📬 Connect with Me  
🔗 **LinkedIn:** [Sunil Karrenolla](https://www.linkedin.com/in/sunil-karrenolla/)  
💻 **GitHub:** [sunilk872](https://github.com/sunilk872/)  
