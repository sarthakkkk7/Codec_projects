# 🚀 Codec Data Analytics internship projects

Welcome to the **Codec_projects** repository! This workspace contains a collection of end-to-end Data Analytics applications focused on e-commerce, retail analytics, and user personalization. 

Each project is containerized within its own directory and includes a standalone Streamlit web application, making it easy to test and visualize the machine learning models in action.

---

## 📁 Repository Structure

### 1. 🛍️ E-Commerce Product Recommendation System
A machine learning-powered recommendation engine that suggests products to users based on their historical behavior. 
* **Technique:** Collaborative Filtering 
* **Algorithm:** Matrix Factorization (Singular Value Decomposition via `scikit-surprise`)
* **Evaluation Metrics:** Precision@K, Recall@K, NDCG@K
* **Use Case:** Predicting user-item interactions and handling personalized product discovery.


### 2. 🛒 Market Basket Analysis Dashboard
An interactive data science dashboard that identifies hidden product associations in transaction data to improve cross-selling and store layout optimization.
* **Technique:** Association Rule Mining
* **Algorithm:** FP-Growth (via `mlxtend`)
* **Evaluation Metrics:** Support, Confidence, Lift
* **Use Case:** Formulating actionable business strategies like product bundling and targeted promotions based on cart data.


---


## 💻 Tech Stack Overview

Across the repository, the primary technologies used include:

* **Python 3.x** for all core logic.
* **Streamlit** for rapid UI prototyping and interactive dashboards.
* **Pandas & NumPy** for data manipulation and synthetic data generation.
* **Scikit-Surprise** for classical matrix factorization.
* **MLxtend** for frequent pattern mining.
* **Plotly** for interactive data visualization.

