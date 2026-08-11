import os
import pandas as pd
import numpy as np
from dbnomics import fetch_series
import torch

os.makedirs('data/rl_macro_test', exist_ok=True)

countries = ["USA", "GBR", "JPN", "DEU", "FRA", "ITA", "CAN", "IND", "BRA", "ZAF", "MEX", "KOR", "AUS", "TUR", "IDN"]

subjects = ["CPALTT01", "PRMNTO01", "SPASTT01", "IRSTCI01", "LRUNTTTT", "XTEXVA01", "XTIMVA01"]

print("Fetching OECD MEI Macro Data...")
try:
    df = fetch_series(
        provider_code='OECD',
        dataset_code='MEI',
        dimensions={
            "LOCATION": countries,
            "SUBJECT": subjects,
            "FREQUENCY": ["M"]
        },
        max_nb_series=1000
    )
    
    df['period'] = pd.to_datetime(df['period'])
    df = df[(df['period'] >= '1990-01-01') & (df['period'] <= '2025-12-31')]
    
    # Check if multiple measures exist and filter to keep one (e.g. IXOB or STSA)
    if 'MEASURE' in df.columns:
        print("Filtering by default MEASURE...")
        df = df[df['MEASURE'].isin(['IXOB', 'STSA', 'VIXOBS'])] # Keep common indices/seasonally adjusted
        
    pivot_df = df.pivot_table(index=['LOCATION', 'period'], columns='SUBJECT', values='value', aggfunc='mean')
    
    # Forward fill then backward fill per country
    pivot_df = pivot_df.groupby('LOCATION').ffill().bfill()
    
    pivot_df.to_csv("data/rl_macro_test/oecd_raw.csv")
    print(f"Saved raw data for {len(pivot_df.index.get_level_values(0).unique())} countries.")
    
    # Normalization: Use strictly 1990-2015 to compute mean/std
    train_mask = pivot_df.index.get_level_values('period') < '2016-01-01'
    train_data = pivot_df[train_mask]
    
    means = train_data.groupby('LOCATION').mean()
    stds = train_data.groupby('LOCATION').std()
    
    normalized_df = (pivot_df - means) / (stds + 1e-8)
    
    # Fill remaining NaNs if a variable was completely empty in the train period
    normalized_df = normalized_df.fillna(0.0)
    
    normalized_df.to_csv("data/rl_macro_test/oecd_normalized.csv")
    print("Saved normalized data.")
    
    tensors = {}
    for country in countries:
        if country in normalized_df.index.get_level_values('LOCATION'):
            c_data = normalized_df.loc[country].sort_index()
            tensors[country] = torch.tensor(c_data.values, dtype=torch.float32)
            
    torch.save(tensors, "data/rl_macro_test/macro_tensors.pt")
    print("Saved PyTorch tensors.")
    
except Exception as e:
    print(f"Failed to fetch data: {e}")
