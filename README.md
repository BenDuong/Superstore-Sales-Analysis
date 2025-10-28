
# Superstore Sales Analysis Project

This project provides a comprehensive analysis of the Superstore Sales dataset from Kaggle. It includes data extraction, transformation, visualization, and interactive tools to explore sales performance.

## 📁 Project Structure

- **Jupyter Notebook (`superstore.ipynb`)**
  - Performs ETL (Extract, Transform, Load) process.
  - Cleans and transforms the dataset.
  - Generates visualizations and statistical insights.

- **Streamlit App (`superstore.py`)**
  - Automatically extracts data from Kaggle.
  - Provides interactive filters and visuals for real-time sales tracking.
  - Enhances user experience with dynamic charts and tables.

- **PowerPoint Presentation (`superstore_sales_analysis_final.pptx`)**
  - Summarizes project objectives, ETL process, insights, and recommendations.

## ⚠️ Important Note

To use the Streamlit app and jupyter notebook, users must input their Kaggle dataset API (vivek468/superstore-dataset-final) and specify the local path to store the data. This is required in the following function:

```python
load_kaggle_data('vivek468/superstore-dataset-final', download_path='Users\Documents')
```

Replace `'Users\Documents'` with your desired local directory.

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Kaggle API credentials
- Required libraries: pandas, streamlit, matplotlib, seaborn, plotly


### Run Streamlit App
```bash
streamlit run app.py
```


## 📊 Features
- Real-time dashboard with interactive filters
- Sales and profit analysis by category, region, and discount
- Visual insights and recommendations


