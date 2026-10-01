import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="StockSense Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
<style>
    .kpi-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    .kpi-title {
        font-size: 18px;
        color: #A0A0A0;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    forecast_df = pd.read_csv('forecast_results.csv')
    inventory_df = pd.read_csv('powerbi_inventory_impact.csv')
    clean_df = pd.read_csv('clean_supply_chain_data.csv', parse_dates=['order_date'])
    return forecast_df, inventory_df, clean_df

forecast_df, inventory_df, clean_df = load_data()

# Sidebar Navigation
st.sidebar.title("📦 StockSense")
st.sidebar.markdown("Demand Forecasting & Inventory Optimization")
page = st.sidebar.radio("Navigation", ["1️⃣ Forecast Accuracy", "2️⃣ Inventory Health", "3️⃣ Business Impact"])

# ==========================================
# PAGE 1: Forecast Accuracy
# ==========================================
if page == "1️⃣ Forecast Accuracy":
    st.title("📈 Forecast Accuracy")
    st.markdown("Monitor how accurately we are predicting demand for our Top Class A products.")
    
    col1, col2 = st.columns(2)
    # Note: Using WAPE% from forecast_results.csv
    avg_wape = forecast_df['WAPE_%'].mean()
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average WAPE (Weighted Error)</div>
            <div class="kpi-value">{avg_wape:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_bias = forecast_df['Bias_Units'].mean()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Bias (Units)</div>
            <div class="kpi-value">{avg_bias:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.subheader("Error Rates by Product (WAPE %)")
    # Sort worst to best
    wape_sorted = forecast_df.sort_values('WAPE_%', ascending=False)
    fig1 = px.bar(wape_sorted, x='WAPE_%', y='Product', orientation='h', 
                  title="WAPE by Product (Higher is worse)", color='WAPE_%', color_continuous_scale='Reds')
    st.plotly_chart(fig1, use_container_width=True)
    
    st.info("💡 **Insight:** Products with a WAPE over 25% are harder to forecast due to high demand volatility (spiky sales). These require a higher Safety Stock buffer.")

# ==========================================
# PAGE 2: Inventory Health
# ==========================================
elif page == "2️⃣ Inventory Health":
    st.title("🏥 Inventory Health & Alerts")
    
    # Calculate Stock Status
    inventory_df['Stock_Status'] = np.where(inventory_df['current_stock'] < inventory_df['Reorder_Point'], '🚨 Reorder Now', '✅ OK')
    
    st.subheader("Current Stock Levels vs Reorder Points")
    
    # Display Matrix/Table
    display_cols = ['Product', 'current_stock', 'Reorder_Point', 'Safety_Stock', 'Stock_Status']
    
    def highlight_reorder(val):
        color = '#ff4b4b' if val == '🚨 Reorder Now' else ''
        return f'background-color: {color}'
    
    st.dataframe(inventory_df[display_cols].style.applymap(highlight_reorder, subset=['Stock_Status']), use_container_width=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = px.bar(inventory_df, x='Product', y='Safety_Stock', title="Required Safety Stock Buffer", color='Safety_Stock', color_continuous_scale='Blues')
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
        
    with col2:
        fig3 = px.bar(inventory_df, x='Product', y=['current_stock', 'Reorder_Point'], barmode='group', title="Current Stock vs Reorder Threshold")
        fig3.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# PAGE 3: Business Impact
# ==========================================
elif page == "3️⃣ Business Impact":
    st.title("💰 Business Impact (ROI)")
    
    total_excess = inventory_df['excess_cost'].sum()
    total_stockout = inventory_df['stockout_cost_avoided'].sum()
    
    st.success(f"**Applying demand-forecast-driven safety stock and reorder policies, this analysis identified ₹{total_excess:,.0f} in excess inventory that could be freed up, and ₹{total_stockout:,.0f} in stockout-driven lost sales that could be avoided, across top SKUs representing the bulk of revenue.**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Excess Inventory Capital Freed</div>
            <div class="kpi-value">₹ {total_excess:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Stockout Losses Avoided</div>
            <div class="kpi-value">₹ {total_stockout:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.subheader("Inventory Inefficiency Breakdown (Naive vs Optimized)")
    
    fig4 = go.Figure(data=[
        go.Bar(name='Naive Stock Policy', x=inventory_df['Product'], y=inventory_df['Naive_Stock_Level'], marker_color='indianred'),
        go.Bar(name='Optimized Stock Policy', x=inventory_df['Product'], y=inventory_df['Optimized_Avg_Stock'], marker_color='lightsalmon')
    ])
    fig4.update_layout(barmode='group', title="Average Stock Units Held: Naive vs Optimized", xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
    
    fig5 = px.bar(inventory_df, x='Product', y='excess_cost', title="Capital Freed per SKU (₹)", text_auto='.2s')
    fig5.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)
