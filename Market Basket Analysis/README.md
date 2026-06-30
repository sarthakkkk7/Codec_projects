# 🛒 Market Basket Analysis Dashboard

An interactive data science dashboard that identifies hidden product associations in transaction data to improve cross-selling, product bundling, and store layout optimization. 

This project utilizes the **FP-Growth Algorithm** to efficiently mine frequent itemsets and extract high-value association rules.

## 🚀 Features
* **FP-Growth Algorithm:** Faster and more memory-efficient than traditional Apriori.
* **Interactive UI:** Dynamically adjust Minimum Support and Minimum Lift thresholds to discover new rules.
* **Data Visualization:** Interactive Plotly scatter plot mapping Support vs. Confidence, sized by Lift.
* **Actionable Insights:** Automatically translates raw mathematical rules into real-world business strategies (e.g., targeted promotions and bundle pricing).

### 📂 Directory Structure
Create a folder named `market-basket-analysis` and arrange your files like this:

```text
market-basket-analysis/
│
├── market_basket_app.py    # The full Streamlit app & pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 💻 Usage

Launch the Streamlit web application. The app will automatically generate a simulated transaction dataset, run the FP-Growth algorithm, and render the dashboard.

```bash
streamlit run market_basket_app.py

```

## 📊 Understanding the Metrics

* **Support:** The frequency of the itemset in the dataset.
* **Confidence:** The likelihood that a customer buys item B, given they have already bought item A.
* **Lift:** The strength of the association. A lift > 1 indicates that the items are frequently bought together because of a distinct relationship, rather than random chance.
