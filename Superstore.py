import os
import io
import zipfile
import pandas as pd
from pandas.core.indexes import category
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.preprocessing import LabelEncoder
import streamlit as st


# Page config
st.set_page_config(page_title="Superstore", page_icon="🏪", layout="wide")

# Title
st.title("📊 Superstore Sales Analysis")
st.markdown("Interactive analysis of superstore sales.")

@st.cache_data
def load_csv_from_bytes(bytes_data):
    return pd.read_csv(io.BytesIO(bytes_data))

@st.cache_data
def load_kaggle_data(dataset_name, download_path, file_type='csv'):
    """
    Downloads and loads a dataset from Kaggle into a pandas DataFrame.
    Tries kaggle API first; if unavailable or fails, looks for an existing CSV file
    in download_path and loads the first matching file.
    """
    # Ensure download directory exists
    if not os.path.exists(download_path):
        try:
            os.makedirs(download_path, exist_ok=True)
        except Exception:
            # If creation fails, continue and attempt reading any absolute path later
            pass

    # Try Kaggle API download if available
    try:
        import kaggle
        try:
            kaggle.api.dataset_download_files(
                dataset_name,
                path=download_path,
                unzip=True
            )
            # If download succeeded, look for csv in download_path
        except Exception as e:
            # Kaggle present but download failed; do fallback to local files
            st.sidebar.warning(f"Kaggle download failed: {e}")
    except Exception:
        # kaggle package not installed / not configured
        st.sidebar.info("kaggle package not available or not configured; trying local files in download path.")

    # After attempting download (or not), search for a CSV/JSON in the download_path
    data_file = None
    if os.path.exists(download_path):
        for root, _, files in os.walk(download_path):
            for f in files:
                if f.lower().endswith(f'.{file_type}'):
                    data_file = os.path.join(root, f)
                    break
            if data_file:
                break

    # If not found, also try some likely filenames in parent folder
    if not data_file:
        likely_names = ['superstore.csv', 'superstore_final.csv', 'superstore-dataset-final.csv']
        for name in likely_names:
            candidate = os.path.join(download_path, name)
            if os.path.exists(candidate):
                data_file = candidate
                break

    if not data_file:
        # Nothing found
        st.sidebar.warning(f"No .{file_type} file found in '{download_path}'.")
        return None

    # Load the found data file into a DataFrame
    try:
        if file_type == 'csv':
            try:
                df = pd.read_csv(data_file)
            except Exception:
                df = pd.read_csv(data_file, encoding="windows-1252", on_bad_lines='skip')
        elif file_type == 'json':
            df = pd.read_json(data_file)
        else:
            st.sidebar.error(f"Unsupported file type: .{file_type}")
            return None
        st.sidebar.success(f"Loaded data from {data_file}")
        return df
    except Exception as e:
        st.sidebar.error(f"Failed to load data file: {e}")
        return None

def convert_date_columns(df: pd.DataFrame):
    for c in df.columns:
        if "date" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
    return df

def add_derived_columns(df: pd.DataFrame):
    if 'Ship Date' in df.columns and 'Order Date' in df.columns:
        df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    if 'Order Date' in df.columns:
        df['Order Day'] = df['Order Date'].dt.day
        df['Order Month'] = df['Order Date'].dt.month
        df['YearMonth'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
    if 'Profit' in df.columns and 'Sales' in df.columns:
        df['Profit Margin'] = np.where(df['Sales'] == 0, np.nan, (df['Profit'] / df['Sales']) * 100)
    return df

def get_mean_and_moe(arr, confidence=0.95):
    arr = np.array(pd.Series(arr).dropna(), dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan
    mean = arr.mean()
    if n < 2:
        return mean, np.nan
    sem = stats.sem(arr)
    ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=sem)
    moe = (ci[1] - ci[0]) / 2
    return mean, moe


def run_ttest(df):
    if 'Segment' in df.columns and 'Profit' in df.columns:
        consumer = df[df['Segment']=='Consumer']['Profit'].dropna()
        corporate = df[df['Segment']=='Corporate']['Profit'].dropna()
        if len(consumer) < 2 or len(corporate) < 2:
            return None, None
        t_stat, p_value = ttest_ind(consumer, corporate, equal_var=False)
        return t_stat, p_value
    return None, None

# --- AUTO LOAD: try to load dataset at start without asking user ---
AUTO_DATASET = 'vivek468/superstore-dataset-final'
AUTO_DOWNLOAD_PATH = r'C:\Users\phuocdh\Documents' ## Change this to your desired path

if 'auto_load_done' not in st.session_state:
    st.session_state['auto_load_done'] = False
    st.session_state['df_auto'] = None
    # attempt automatic load
    df_auto = load_kaggle_data(AUTO_DATASET, download_path=AUTO_DOWNLOAD_PATH, file_type='csv')
    if df_auto is not None:
        st.session_state['df_auto'] = df_auto
        st.session_state['auto_load_done'] = True
    else:
        st.session_state['auto_load_done'] = True  # attempted but not found

# Primary dataframe variable: prefer auto-loaded data, else allow upload/manual
df = st.session_state.get('df_auto', None)



if df is None:
    st.info("No data loaded yet. Upload a CSV or enable manual Kaggle download.")
    st.stop()

# preprocess
df = convert_date_columns(df)
df = add_derived_columns(df)

# Sidebar filters (Region, Category, Profit range, Profit Margin)
st.sidebar.header("Filters")

#Region filter
selected_region = st.sidebar.multiselect(
    "Select Region",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

#Category Filter
selected_category = st.sidebar.multiselect(
    "Select Category",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

#Order Date Filter

order_range = st.sidebar.slider(
    "Order Date Range",
    min_value=df['Order Date'].min().date(),
    max_value=df['Order Date'].max().date(),
    value=(df['Order Date'].min().date(), df['Order Date'].max().date()),  # Initial range covering all dates
    format="DD/MMM/YYYY"
)

#Profit Filter
profit_range = st.sidebar.slider(
    "Profit",
    min_value=int(df['Profit'].min()),
    max_value=int(df['Profit'].max()),
    value=(0, 5000)
)

#Profit Margin
pm_range = st.sidebar.slider(
    "Profit Margin (%)",
    min_value=int(df['Profit Margin'].min()),
    max_value=int(df['Profit Margin'].max()),
    value=(0, 10),
    format="%d%%"
)



# Apply filters
df_filtered = df.copy()
if selected_region != 'All' and 'Region' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Region'].isin(selected_region)]
if selected_category != 'All' and 'Category' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Category'].isin(selected_category)]
if 'Order Date' in df_filtered.columns and isinstance(order_range, tuple):
    df_filtered['Order Date'] = df_filtered['Order Date'].dt.date
    df_filtered = df_filtered[df_filtered['Order Date'].between(order_range[0], order_range[1])]
if 'Profit' in df_filtered.columns and isinstance(profit_range, tuple):
    df_filtered = df_filtered[df_filtered['Profit'].between(float(profit_range[0]), float(profit_range[1]))]
if pm_range is not None and 'Profit Margin' in df_filtered.columns and isinstance(pm_range, tuple):
    df_filtered = df_filtered[df_filtered['Profit Margin'].between(float(pm_range[0]), float(pm_range[1]))]


# Top-row metrics in compact columns to save space
st.header("Overview")
m1, m2, m3 = st.columns(3)

with m1:
      total_sales = df_filtered['Sales'].sum()
      st.metric(
        "Total Sales 📈",
        f"${total_sales:.0f}",
        f"{total_sales - df['Sales'].sum():+.1f} vs all"
    )

with m2:
      total_profit = df_filtered['Profit'].sum()
      st.metric(
        "Total Profit💲",
        f"${total_profit:.0f}",
        f"{total_profit - df['Profit'].sum():+.1f} vs all"
    )

with m3:
      avg_margin = df_filtered['Profit Margin'].mean()
      st.metric(
        "Avg Profit Margin",
        f"{avg_margin :.1f}%"
    )
      



# Charts
col1, col2 = st.columns(2)
# Boxplot in col1
with col1:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_filtered, x='Segment', y='Profit', ax=ax,palette="Set2")
    ax.set_xlabel('Segment')
    ax.set_ylabel('Profit')
    ax.set_title('Profit by Segments')
    plt.tight_layout()
    st.pyplot(fig)

# Line chart in col2
with col2:

    cat_profit = df_filtered.groupby('Sub-Category', dropna=False).agg(
        profit=('Profit', 'mean'),
        profit_margin=('Profit Margin', 'mean')
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=cat_profit,  x='profit', y='profit_margin',hue='Sub-Category', palette="Set2")
    ax.set_xlabel("Mean Profit")
    ax.set_ylabel("Mean Profit Margin (%)")
    ax.set_title("Sub-Category: Mean Profit vs Mean Profit Margin")
    plt.tight_layout()
    st.pyplot(fig)

col1_r1, col2_r2 = st.columns(2)
with col1_r1:
# Time series plot
  grouped_df = df_filtered.groupby('YearMonth')['Sales'].sum().reset_index()
  grouped_df_mean = grouped_df['Sales'].mean()

  fig, ax = plt.subplots(figsize=(8,6)) 
  ax.plot(grouped_df['YearMonth'], grouped_df['Sales'], marker='o')
  ax.axhline(y=grouped_df_mean, color='orange', linestyle='--')
  ax.set_title('Total Sales Over Time by Category', fontsize=12)
  ax.set_xlabel('Order Month', fontsize=8)
  ax.legend(['Monthly Sales', 'Overall Avg Sales'])
  plt.xticks(rotation=45, fontsize=8)
  plt.tight_layout()
  st.pyplot(fig)


with col2_r2:
 ship_avg = df_filtered.groupby('YearMonth')['Shipping Days'].mean().reset_index()
 ship_avg_mean = ship_avg['Shipping Days'].mean()
 fig, ax = plt.subplots(figsize=(8,6))
 ax.plot(ship_avg['YearMonth'], ship_avg['Shipping Days'], marker='o')
 ax.axhline(y=ship_avg_mean, color='orange', linestyle='--')
 ax.set_xlabel('YearMonth', fontsize=8)
 ax.set_ylabel('Avg Shipping Days', fontsize=8)
 ax.set_title('Average Shipping Days by Month', fontsize=12)
 ax.legend(['Avg Shipping Days', 'Overall Avg Shipping Days'])
 plt.xticks(rotation=45, fontsize=8)
 plt.tight_layout()
 st.pyplot(fig)


# Category mean and margin of error (compact table)
st.subheader("Sub-Category mean profit ± margin of error (95%)")
if 'Category' in df_filtered.columns and 'Profit' in df_filtered.columns:
    grouped = df_filtered.groupby('Sub-Category')['Profit']
    rows = []
    for name, group in grouped:
        mean, moe = get_mean_and_moe(group)
        formatted = f"{mean:.2f} ± {moe:.2f}" if not np.isnan(moe) else f"{mean:.2f} ± N/A"
        rows.append({"Category": name, "mean": mean, "moe": moe, "formatted": formatted})
    st.table(pd.DataFrame(rows).sort_values("Category").reset_index(drop=True))
else:
    st.info("Sub-Category or Profit column missing.")

# T-test
st.subheader("T-test: Consumer vs Corporate (Profit)")
t_stat, p_value = run_ttest(df_filtered)
if t_stat is None:
    st.info("Not enough data for t-test or required columns missing.")
else:
    st.write(f"t-statistic: {t_stat:.4f}")
    st.write(f"p-value: {p_value:.6f}")
    if p_value < 0.05:
        st.write("Conclusion: Significant difference in average profit between Consumer and Corporate (p < 0.05).")
    else:
        st.write("Conclusion: No significant difference detected (p >= 0.05).")

# Quick data snapshot
st.subheader("Filtered Data")
df_show = df_filtered[['Order ID', 'Order Date', 'Ship Date', 'Customer Name', 'Segment', 'Category', 'Sub-Category','Product Name', 'Sales', 'Profit','Discount','Profit Margin']]
st.dataframe(df_show)

# Download filtered data
csv_bytes = df_show.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered CSV", data=csv_bytes, file_name="superstore_filtered.csv", mime="text/csv")

st.markdown("Done.")