import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

print("--- STARTING DIAGNOSTIC & FIX ---")

# 1. Load data
df = pd.read_csv('clean_supply_chain_data.csv', parse_dates=['order_date'])
df['product_name'] = df['product_name'].str.strip()

# Print data drop-off analysis
print("\n[DIAGNOSTIC] Monthly Order Volume:")
monthly_counts = df['order_date'].dt.to_period('M').value_counts().sort_index()
print(monthly_counts.tail(10))
print("\n[DIAGNOSTIC] The data drops from ~5300 to ~2100 in Oct 2017. This 'dead tail' corrupted the test set.")

# 2. Exclude dead tail (keep data before Oct 2017)
df_clean = df[df['order_date'] < '2017-10-01']

# Re-run logic on clean data
revenue_by_prod = df_clean.groupby('product_name')['Order Item Total'].sum().sort_values(ascending=False)
top_20_products = revenue_by_prod.head(20).index.tolist()

total_revenue = revenue_by_prod.sum()
revenue_cum_pct = revenue_by_prod.cumsum() / total_revenue
abc_mapping = {prod: 'A' if pct <= 0.80 else 'B' if pct <= 0.95 else 'C' for prod, pct in revenue_cum_pct.items()}

df_top20 = df_clean[df_clean['product_name'].isin(top_20_products)]
lead_time_dict = df_top20.groupby('product_name')['actual_lead_time'].mean().to_dict()
unit_cost_dict = df_top20.groupby('product_name')['Order Item Product Price'].mean().to_dict()

forecast_results_list = []
inventory_summary_list = []
monthly_demand_list = []
backtest_results_list = []

service_level_z = 1.645 # 95%
holding_cost_rate = 0.20
exchange_rate = 83

for prod in top_20_products:
    prod_df = df_top20[df_top20['product_name'] == prod]
    
    # 2. Monthly Forecasting
    monthly_data = prod_df.set_index('order_date').resample('ME')['quantity_sold'].sum().reset_index()
    monthly_data['month'] = monthly_data['order_date'].dt.to_period('M')
    
    # Train-test split (hold out last 6 months of CLEAN data: Apr-Sep 2017)
    train = monthly_data.iloc[:-6]
    test = monthly_data.iloc[-6:]
    
    if len(train) == 0:
        model_name = 'Naive Mean'
        next_month_forecast = monthly_data['quantity_sold'].mean()
        preds = np.repeat(next_month_forecast, len(test))
    elif len(train) > 12:
        model_name = 'Holt-Winters'
        model = ExponentialSmoothing(train['quantity_sold'], trend='add', seasonal=None, initialization_method="estimated").fit()
        preds = model.forecast(len(test)).values
        next_month_forecast = model.forecast(1).values[0]
    else:
        model_name = 'Simple Exp Smoothing'
        model = ExponentialSmoothing(train['quantity_sold'], trend=None, seasonal=None, initialization_method="estimated").fit()
        preds = model.forecast(len(test)).values
        next_month_forecast = model.forecast(1).values[0]
    
    preds = np.clip(preds, a_min=0, a_max=None)
    
    actuals = test['quantity_sold'].values
    forecasts = preds
    bias_units = np.sum(forecasts - actuals)
    wape = np.sum(np.abs(forecasts - actuals)) / np.sum(actuals) if np.sum(actuals) > 0 else 0
    next_month_forecast = max(0, next_month_forecast)
    
    for idx in range(len(test)):
        monthly_demand_list.append({
            'product': prod,
            'month': str(test['month'].iloc[idx]),
            'actual': actuals[idx],
            'forecast': forecasts[idx]
        })
        
    forecast_results_list.append({
        'product': prod,
        'model_used': model_name,
        'wape': wape,
        'bias_units': bias_units,
        'next_month_forecast': next_month_forecast
    })
    
    # 3. Inventory Logic (Daily based)
    daily_data = prod_df.set_index('order_date').resample('D')['quantity_sold'].sum().fillna(0)
    
    avg_daily_demand = daily_data.mean()
    std_daily_demand = daily_data.std()
    
    avg_lead_time = lead_time_dict[prod]
    unit_cost_usd = unit_cost_dict[prod]
    
    # Safety Stock
    lead_time_std = 0
    safety_stock = service_level_z * np.sqrt((avg_lead_time * std_daily_demand**2) + (avg_daily_demand**2 * lead_time_std**2))
    reorder_point = (avg_daily_demand * avg_lead_time) + safety_stock
    
    # EOQ
    order_cost = 50 
    annual_demand = avg_daily_demand * 365
    eoq = np.sqrt((2 * annual_demand * order_cost) / (unit_cost_usd * holding_cost_rate))
    
    optimized_stock_level = safety_stock + (eoq / 2)
    
    # Naive Baseline uses a HEALTHY last month's demand now
    last_month_demand = test['quantity_sold'].iloc[-1] if not test.empty else 0
    naive_stock_level = last_month_demand * 1.5
    
    inventory_summary_list.append({
        'product': prod,
        'abc_category': abc_mapping[prod],
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'eoq': eoq,
        'naive_stock_level': naive_stock_level,
        'optimized_stock_level': optimized_stock_level,
        'unit_price': unit_cost_usd * exchange_rate
    })

    # 4. Backtesting (Last 180 Days of clean data)
    backtest_data = daily_data.iloc[-180:]
    
    def simulate_inventory(policy_type, rop, order_qty):
        stock = rop + order_qty
        pending_order = 0
        days_to_delivery = 0
        
        stockout_days = 0
        total_demand = 0
        
        for demand in backtest_data.values:
            total_demand += demand
            if days_to_delivery == 0 and pending_order > 0:
                stock += pending_order
                pending_order = 0
                
            if stock >= demand:
                stock -= demand
            else:
                stock = 0
                stockout_days += 1
                
            if policy_type == 'naive':
                if stock < rop and pending_order == 0:
                    pending_order = rop - stock
                    days_to_delivery = int(avg_lead_time)
            elif policy_type == 'optimized':
                if stock < rop and pending_order == 0:
                    pending_order = order_qty
                    days_to_delivery = int(avg_lead_time)
                    
            if days_to_delivery > 0:
                days_to_delivery -= 1
                
        return stockout_days
        
    naive_so_days = simulate_inventory('naive', naive_stock_level, 0)
    opt_so_days = simulate_inventory('optimized', reorder_point, eoq)
    
    backtest_results_list.append({'product': prod, 'policy': 'naive', 'stockout_days': naive_so_days})
    backtest_results_list.append({'product': prod, 'policy': 'optimized', 'stockout_days': opt_so_days})

# Compile DataFrames
forecast_df = pd.DataFrame(forecast_results_list)
inventory_df = pd.DataFrame(inventory_summary_list)
backtest_df = pd.DataFrame(backtest_results_list)

# Compute excess cost correctly
inventory_df['unit_cost_inr'] = inventory_df['unit_price'] * 0.70 # Assume 30% margin
inventory_df['excess_cost'] = np.maximum(0, inventory_df['naive_stock_level'] - inventory_df['optimized_stock_level']) * inventory_df['unit_cost_inr']

# Determine Stock Status counts
def get_status(row):
    if row['naive_stock_level'] <= row['reorder_point']:
        return "Reorder Now"
    elif row['naive_stock_level'] > (row['reorder_point'] + row['eoq']):
        return "Overstocked"
    else:
        return "OK"
        
inventory_df['Stock Status'] = inventory_df.apply(get_status, axis=1)
status_counts = inventory_df['Stock Status'].value_counts().reset_index()
status_counts.columns = ['Status', 'Count']
status_counts.to_csv('stock_status_counts.csv', index=False)

# Export V2 CSVs
inventory_df[['product', 'abc_category', 'safety_stock', 'reorder_point', 'eoq', 'naive_stock_level', 'optimized_stock_level', 'excess_cost', 'Stock Status']].to_csv('inventory_summary_v2.csv', index=False)
forecast_df.to_csv('forecast_results_v2.csv', index=False)
pd.DataFrame(monthly_demand_list).to_csv('monthly_demand_v2.csv', index=False)
backtest_df.to_csv('backtest_results_v2.csv', index=False)

# Diagnostics
print(f"\n[RESULTS] NEW Avg WAPE: {forecast_df['wape'].mean():.2%}")
print(f"[RESULTS] NEW Total Bias: {forecast_df['bias_units'].sum():.0f} units")
print(f"[RESULTS] NEW Excess Capital Freed: Rs {inventory_df['excess_cost'].sum():,.0f}")
print(f"[RESULTS] NEW Naive Stockout Days: {backtest_df[backtest_df['policy'] == 'naive']['stockout_days'].sum()}")
print(f"[RESULTS] NEW Optimized Stockout Days: {backtest_df[backtest_df['policy'] == 'optimized']['stockout_days'].sum()}")
print("\n[RESULTS] Stock Status Counts on clean data:")
print(inventory_df['Stock Status'].value_counts())
