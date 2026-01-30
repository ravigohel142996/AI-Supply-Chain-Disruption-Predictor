"""Generate sample supply chain data for demonstration."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sample_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic supply chain data.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sample data
    """
    np.random.seed(seed)
    
    # Generate base features
    order_ids = [f"ORD{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
    
    # Generate order dates
    start_date = datetime.now() - timedelta(days=365)
    order_dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
    
    # Numeric features
    order_values = np.random.lognormal(mean=8.5, sigma=0.5, size=n_samples)
    shipping_distances = np.random.gamma(shape=2, scale=200, size=n_samples)
    lead_times = np.random.gamma(shape=3, scale=2, size=n_samples)
    supplier_reliability = np.random.beta(a=8, b=2, size=n_samples)
    inventory_levels = np.random.gamma(shape=5, scale=100, size=n_samples)
    demand_forecasts = np.random.gamma(shape=4, scale=120, size=n_samples)
    weather_risk = np.random.beta(a=2, b=5, size=n_samples)
    
    # Categorical features
    shipping_modes = np.random.choice(['Air', 'Sea', 'Road', 'Rail'], size=n_samples, p=[0.15, 0.25, 0.45, 0.15])
    supplier_regions = np.random.choice(['Asia', 'Europe', 'North America', 'South America'], 
                                       size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    product_categories = np.random.choice(['Electronics', 'Automotive', 'Textiles', 'Food', 'Chemicals'],
                                         size=n_samples, p=[0.25, 0.2, 0.2, 0.2, 0.15])
    seasons = np.array([date.month for date in order_dates])
    seasons = pd.cut(seasons, bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])
    carriers = np.random.choice(['Carrier_A', 'Carrier_B', 'Carrier_C', 'Carrier_D'],
                               size=n_samples, p=[0.3, 0.25, 0.25, 0.2])
    
    # Generate target variable (is_delayed) with realistic dependencies
    delay_prob = (
        0.1 +
        0.3 * (1 - supplier_reliability) +
        0.2 * weather_risk +
        0.15 * (lead_times > lead_times.mean()).astype(int) +
        0.1 * (shipping_distances > shipping_distances.mean()).astype(int) +
        0.15 * (shipping_modes == 'Sea').astype(int)
    )
    delay_prob = np.clip(delay_prob, 0, 1)
    is_delayed = (np.random.random(n_samples) < delay_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'order_id': order_ids,
        'order_date': order_dates,
        'order_value': order_values,
        'shipping_distance': shipping_distances,
        'lead_time': lead_times,
        'supplier_reliability_score': supplier_reliability,
        'inventory_level': inventory_levels,
        'demand_forecast': demand_forecasts,
        'weather_risk_index': weather_risk,
        'shipping_mode': shipping_modes,
        'supplier_region': supplier_regions,
        'product_category': product_categories,
        'season': seasons,
        'carrier': carriers,
        'is_delayed': is_delayed
    })
    
    return df


def save_sample_data(output_path: str = 'data/sample/supply_chain_data.csv', n_samples: int = 1000):
    """
    Generate and save sample data.
    
    Args:
        output_path: Path to save the data
        n_samples: Number of samples to generate
    """
    df = generate_sample_data(n_samples)
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Delay rate: {df['is_delayed'].mean():.2%}")


if __name__ == "__main__":
    save_sample_data()
