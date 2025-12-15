"""
Data Conversion Utility for Bank Stock Data
=============================================
Converts Bank stock data from Vietnamese format (VND) to standardized format
with optional USD conversion.

Features:
  - Clean and transform Vietnamese stock data (Bank_realdata.csv)
  - Standardize column names and formats
  - Convert prices from VND to USD using exchange rate
"""

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

USD_RATE = 26319.95  # 1 USD = 26319.95 VND


# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================

def clean_vietnamese_bank_data(filepath: str) -> pd.DataFrame:
    """
    Clean and transform Vietnamese bank stock data to standard format.

    Handles:
      - Price column cleaning (remove commas, convert to float)
      - Volume column cleaning (remove 'M' suffix, convert to actual volume)
      - Date conversion from DD/MM/YYYY to YYYY-MM-DD
      - Column renaming to standard format
      - Symbol column addition

    Parameters:
    -----------
    filepath : str
        Path to the Vietnamese format CSV file

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with columns: time, open, high, low, close, volume, symbol
    """
    # Load the data
    df = pd.read_csv(filepath)

    # 1. Clean price columns (remove commas, convert to float)
    price_cols = ['Lần cuối', 'Mở', 'Cao', 'Thấp']
    for col in price_cols:
        df[col] = df[col].str.replace(',', '', regex=False).astype(float)

    # 2. Clean volume column (remove 'M'/'K' suffix, convert to actual volume)
    # Check if suffix is 'M' (millions) or 'K' (thousands)
    def convert_volume(volume_str):
        """Convert volume string with M/K suffix to numeric value."""
        if pd.isna(volume_str):
            return 0

        volume_str = str(volume_str).strip()

        if volume_str.endswith('M'):
            # Millions: multiply by 1,000,000
            return float(volume_str.replace('M', '')) * 1000000
        elif volume_str.endswith('K'):
            # Thousands: multiply by 1,000
            return float(volume_str.replace('K', '')) * 1000
        else:
            # No suffix, assume it's already in actual units
            return float(volume_str)

    df['KL'] = df['KL'].apply(convert_volume).astype(np.int64)

    # 3. Convert date format from DD/MM/YYYY to YYYY-MM-DD
    df['time'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

    # 4. Rename columns to standard format
    df = df.rename(columns={
        'Mở': 'open',
        'Cao': 'high',
        'Thấp': 'low',
        'Lần cuối': 'close',
        'KL': 'volume'
    })

    # 5. Add symbol column
    df['symbol'] = 'BANK'

    # 6. Select and reorder columns
    df_cleaned = df[['time', 'open', 'high', 'low', 'close', 'volume', 'symbol']].copy()

    return df_cleaned


def convert_vnd_to_usd(df: pd.DataFrame,
						usd_rate: float = 26319.95,
						price_cols: list = None,
						volume_multiplier: float = 1000.0) -> pd.DataFrame:
    """
    Convert OHLCV values from VND to USD.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: open, high, low, close, volume

    usd_rate : float
        Exchange rate (VND per 1 USD). Default: 26319.95 VND/USD

    price_cols : list, optional
        List of price columns to convert. Default: ['open', 'high', 'low', 'close']

    volume_multiplier : float
        Multiplier to convert volume from "thousands" to actual units. Default: 1000.0

    Returns:
    --------
    pd.DataFrame
        DataFrame with converted OHLCV values in USD
    """
    # Make a copy to avoid modifying original
    df_converted = df.copy()

    # Default price columns if not specified
    if price_cols is None:
        price_cols = ['open', 'high', 'low', 'close']

    # Validate required columns exist
    required_cols = price_cols + ['volume']
    missing_cols = [col for col in required_cols if col not in df_converted.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Convert price columns from VND to USD
    for col in price_cols:
        df_converted[col] = (df_converted[col] / usd_rate).round(4)

    # Convert volume from thousands to actual volume in USD value
    df_converted['volume'] = (df_converted['volume'] * volume_multiplier / usd_rate).round(2)

    return df_converted


def print_conversion_summary(df_original: pd.DataFrame, df_converted: pd.DataFrame,
                            usd_rate: float, filename: str):
    """Print summary statistics of the conversion."""
    print("\n" + "="*70)
    print("CONVERSION SUMMARY".center(70))
    print("="*70)
    print(f"Exchange Rate: 1 USD = {usd_rate} VND")
    print(f"Rows converted: {len(df_converted)}")
    print(f"Date range: {df_converted['time'].min()} to {df_converted['time'].max()}")
    print(f"Output file: {filename}")

    print(f"\n{'Price Range (USD):':<30}")
    print(f"  Open:   {df_converted['open'].min():.4f} - {df_converted['open'].max():.4f} USD")
    print(f"  Close:  {df_converted['close'].min():.4f} - {df_converted['close'].max():.4f} USD")
    print(f"  High:   {df_converted['high'].min():.4f} - {df_converted['high'].max():.4f} USD")
    print(f"  Low:    {df_converted['low'].min():.4f} - {df_converted['low'].max():.4f} USD")

    print(f"\n{'Volume Range (USD):':<30}")
    print(f"  Min: {df_converted['volume'].min():,.2f} USD")
    print(f"  Max: {df_converted['volume'].max():,.2f} USD")

    print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = 'Bank_realdata2.csv'  # Change to 'Bank_train.csv' as needed
    OUTPUT_CLEANED = f'{INPUT_FILE.replace(".csv", "")}_cleaned.csv'
    OUTPUT_USD = f'{INPUT_FILE.replace(".csv", "")}_USD.csv'

    print(f"Processing: {INPUT_FILE}")
    print("-" * 70)

    # ---- Step 1: Clean Vietnamese format data ----
    print("\n[Step 1] Cleaning Vietnamese format data...")
    try:
        df_cleaned = clean_vietnamese_bank_data(INPUT_FILE)
        df_cleaned.to_csv(OUTPUT_CLEANED, index=False)
        print(f"✓ Cleaned data saved to: {OUTPUT_CLEANED}")
        print(f"  Shape: {df_cleaned.shape}")
        print(f"\nFirst 3 rows:")
        print(df_cleaned.head(3))
    except Exception as e:
        print(f"✗ Error cleaning data: {e}")
        exit(1)

    # ---- Step 2: Convert to USD ----
    print(f"\n[Step 2] Converting from VND to USD (rate: {USD_RATE})...")
    try:
        df_usd = convert_vnd_to_usd(
            df=df_cleaned,
            usd_rate=USD_RATE,
            price_cols=['open', 'high', 'low', 'close'],
            volume_multiplier=1000.0
        )
        df_usd.to_csv(OUTPUT_USD, index=False)
        print(f"✓ USD converted data saved to: {OUTPUT_USD}")
        print(f"  Shape: {df_usd.shape}")
        print(f"\nFirst 3 rows (USD):")
        print(df_usd.head(3))
    except Exception as e:
        print(f"✗ Error converting to USD: {e}")
        exit(1)

    # ---- Step 2.5: Sort by time (oldest to newest) ----
    print(f"\n[Step 2.5] Sorting data by time (oldest to newest)...")
    try:
        # Convert time to datetime for proper sorting
        df_usd['time'] = pd.to_datetime(df_usd['time'])
        # Sort in ascending order (oldest first)
        df_usd = df_usd.sort_values('time').reset_index(drop=True)
        # Convert back to string format
        df_usd['time'] = df_usd['time'].dt.strftime('%Y-%m-%d')
        # Save sorted data
        df_usd.to_csv(OUTPUT_USD, index=False)
        print(f"✓ Data sorted and saved to: {OUTPUT_USD}")
        print(f"  Date range: {df_usd['time'].min()} → {df_usd['time'].max()}")
        print(f"  Total rows: {len(df_usd)}")
    except Exception as e:
        print(f"✗ Error sorting data: {e}")
        exit(1)

    # ---- Step 3: Print summary ----
    print_conversion_summary(df_cleaned, df_usd, USD_RATE, OUTPUT_USD)

    print("\n✓ All conversions completed successfully!")