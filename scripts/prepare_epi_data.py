import torch
import pandas as pd
import numpy as np
import os
import io
import urllib.request

def prepare_epi_data():
    print("Downloading JHU COVID-19 US Confirmed Cases...")
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    
    import os
    os.system(f"curl -s {url} -o covid.csv")
    df = pd.read_csv("covid.csv")
        
    print(f"Downloaded shape: {df.shape}")
    
    # Filter out unassigned or out of state
    df = df[df['Admin2'].notna()]
    df = df[~df['Admin2'].str.contains("Unassigned|Out of")]
    
    # Dates start from column 11
    date_cols = df.columns[11:]
    
    # We want daily NEW cases, not cumulative
    # To smooth out reporting artifacts (e.g., zero on weekends, double on Monday), we'll use a 7-day rolling average
    
    states = df['Province_State'].unique()
    
    valid_states = []
    epi_tensors = {}
    
    for state in states:
        state_df = df[df['Province_State'] == state]
        
        # Need at least 10 counties
        if len(state_df) < 10:
            continue
            
        # Select the 10 most populated counties (proxy by total cases at the end)
        state_df = state_df.sort_values(by=date_cols[-1], ascending=False).head(10)
        
        # Get raw cumulative time series
        ts = state_df[date_cols].values # (10, num_days)
        
        # Daily new cases
        new_cases = np.diff(ts, axis=1) # (10, num_days - 1)
        
        # Smooth with 7-day moving average
        smoothed = np.zeros_like(new_cases, dtype=np.float32)
        for i in range(10):
            smoothed[i] = pd.Series(new_cases[i]).rolling(window=7, min_periods=1).mean().values
            
        # Causal Standardize each county using a 30-day trailing window to prevent future leakage
        for i in range(10):
            s = pd.Series(smoothed[i])
            rolling_mean = s.rolling(window=30, min_periods=1).mean()
            rolling_std = s.rolling(window=30, min_periods=2).std()
            rolling_std = rolling_std.fillna(1.0)
            rolling_std[rolling_std == 0] = 1.0
            smoothed[i] = ((s - rolling_mean) / rolling_std).values
                
        # Shape to (num_days - 1, 10)
        smoothed = smoothed.T
        
        if smoothed.shape[0] > 500:
            epi_tensors[state] = torch.tensor(smoothed, dtype=torch.float32)
            valid_states.append(state)
            
        if len(valid_states) == 15:
            break
            
    print(f"Selected {len(valid_states)} states with 10 counties each.")
    
    os.makedirs('data/rl_macro_test', exist_ok=True)
    torch.save(epi_tensors, "data/rl_macro_test/epi_tensors.pt")
    print("Saved to data/rl_macro_test/epi_tensors.pt")
    
    for k, v in epi_tensors.items():
        print(f" {k}: {v.shape}")

if __name__ == "__main__":
    prepare_epi_data()
