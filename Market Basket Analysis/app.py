import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import plotly.express as px

# ==========================================
# 1. GENERATE DUMMY TRANSACTION DATA
# ==========================================
@st.cache_data
def load_transaction_data():
    np.random.seed(42)
    # Define a small grocery catalog
    catalog = ['Milk', 'Bread', 'Butter', 'Eggs', 'Diapers', 'Beer', 'Apples', 'Coffee', 'Tea', 'Sugar']
    
    # Generate 500 random transactions
    transactions = []
    for _ in range(500):
        # A transaction has 2 to 6 random items
        basket_size = np.random.randint(2, 7)
        basket = list(np.random.choice(catalog, size=basket_size, replace=False))
        
        # Inject intentional associations to make the data interesting
        if 'Diapers' in basket and np.random.random() > 0.3:
            if 'Beer' not in basket: basket.append('Beer')
        if 'Milk' in basket and np.random.random() > 0.4:
            if 'Bread' not in basket: basket.append('Bread')
        if 'Coffee' in basket and np.random.random() > 0.5:
            if 'Sugar' not in basket: basket.append('Sugar')
            
        transactions.append(basket)
    return transactions

# ==========================================
# 2. RUN MARKET BASKET ANALYSIS
# ==========================================
@st.cache_data
def run_mba(transactions, min_support=0.05, min_threshold=1.0, metric="lift"):
    # Convert lists of items into a One-Hot Encoded boolean DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apply FP-Growth
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    
    # Ensure itemsets column contains frozensets of strings (not numpy types)
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(
        lambda x: frozenset(str(item) for item in x)
    )
    
    # Generate Association Rules
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    
    # Clean up columns for display
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules = rules.sort_values(by='lift', ascending=False)
    
    return frequent_itemsets, rules

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Market Basket Analysis", layout="wide")

st.title("🛒 Market Basket Analysis Dashboard")
st.write("Using FP-Growth to identify product associations for cross-selling.")
st.markdown("---")

# Load Data
transactions = load_transaction_data()

# Sidebar configuration
st.sidebar.header("Algorithm Settings")
min_supp = st.sidebar.slider("Minimum Support", 0.01, 0.20, 0.05, 0.01, 
                             help="Minimum frequency of the itemset in the dataset.")
min_lift = st.sidebar.slider("Minimum Lift", 0.5, 3.0, 1.0, 0.1, 
                             help="Lift > 1 implies a positive association.")

# Run Analysis
frequent_itemsets, rules = run_mba(transactions, min_support=min_supp, min_threshold=min_lift, metric="lift")

# Layout: Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Top Association Rules")
    st.write(f"Found **{len(rules)}** rules based on your filters.")
    
    # Display table formatted nicely
    display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    display_rules.rename(columns={'antecedents': 'If Bought', 'consequents': 'Then Buys'}, inplace=True)
    st.dataframe(display_rules.head(10).style.background_gradient(cmap='Greens', subset=['lift']))

with col2:
    st.subheader("📈 Rule Visualization (Support vs Confidence)")
    if not rules.empty:
        # Scatter plot of rules
        fig = px.scatter(
            rules, 
            x="support", 
            y="confidence", 
            size="lift", 
            color="lift",
            hover_data=['antecedents', 'consequents'],
            labels={"support": "Support (Frequency)", "confidence": "Confidence (Likelihood)"},
            title="Association Rules Bubble Chart",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No rules found. Try lowering the Support or Lift thresholds.")

st.markdown("---")

# ==========================================
# 4. ACTIONABLE BUSINESS INSIGHTS
# ==========================================
st.subheader("💡 Actionable Business Insights")

if not rules.empty:
    top_rule = rules.iloc[0]
    ant = top_rule['antecedents']
    con = top_rule['consequents']
    lift_val = top_rule['lift']
    conf_val = top_rule['confidence'] * 100
    
    st.success(f"**Primary Discovery:** Customers who buy **{ant}** are heavily inclined to buy **{con}**.")
    
    st.markdown(f"""
    Based on the highest-lift association rules extracted from the data, here are three strategies to deploy immediately:
    
    * **Cross-Selling / Bundling:** Combine **{ant}** and **{con}** into a slight discount bundle. Because the confidence is {conf_val:.1f}% and lift is {lift_val:.2f}, grouping them will accelerate volume sales for both items.
    * **Store Layout Optimization:** If this is a physical store, place **{ant}** and **{con}** at opposite ends of an aisle (or the store). This forces the customer to walk past other items, increasing the chance of impulse buys along the way.
    * **Targeted Promotions:** If a customer adds **{ant}** to their online cart, instantly trigger a recommendation pop-up for **{con}**. Do not discount **{con}**—the high confidence implies they are likely to buy it at full price anyway if prompted.
    """)
else:
    st.info("Adjust the sliders on the left to discover actionable rules.")