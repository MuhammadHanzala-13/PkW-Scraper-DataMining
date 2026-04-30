# Pakistan Used Cars Price Prediction: An End-to-End Data Mining Project 🚗

This repository serves as the complete documentation and source code for the **Used Car Price Prediction** project, designed specifically to meet the requirements of university-level data mining and exploratory data analysis.

---

## 1. Introduction and Target Problem
**Target Problem:** The used car market in Pakistan is highly volatile and disorganized. Buyers and sellers struggle to determine the "fair market value" of a vehicle due to rapid inflation, undocumented depreciation, and arbitrary pricing by dealers.

**Objective:** To develop a robust data mining pipeline that acquires real-world automotive data, discovers hidden depreciation trends, and applies machine learning algorithms to accurately estimate the market value of a vehicle based on its technical specifications.

## 2. Short Literature Review (Guidelines for PPT)
*(For your presentation, you can look up and summarize 2-3 papers on Google Scholar based on these themes)*
- **Paper 1 Idea:** A study on how machine learning algorithms (like Random Forest and XGBoost) outperform traditional statistical models in predicting used car prices by capturing non-linear relationships (like the sudden drop in price after 100,000 km).
- **Paper 2 Idea:** Research on the impact of "Feature Engineering" (e.g., calculating exact car age instead of just the manufacturing year) on the accuracy of predictive regression models in automotive markets.
- **Paper 3 Idea:** Exploring the usage of web scraping techniques to build localized, real-time datasets for market sectors that lack open-source datasets (like the Pakistani auto market).

## 3. Data Acquisition & Preprocessing (Open Ended Lab 01)
To ensure the model learns from realistic, localized data, we avoided generic Kaggle datasets and built a custom pipeline:
- **Web Scraping (`pakwheels_scraper.py` & `pakwheels_enricher.py`):** Utilized high-concurrency parsing with BeautifulSoup to extract thousands of live listings from PakWheels.com.
- **Data Preprocessing (`pakwheels_data_engineering.py`):** 
  - Converted human-readable currency (Lacs/Crores) into pure mathematical integers.
  - Handled missing values (imputation) and dropped statistical outliers.
  - Extracted specific `brand` and `model` (e.g., Civic, Corolla) directly from unstructured text titles.

## 4. Feature Engineering & Exploratory Data Analysis (EDA)
Before modeling, the data was mathematically transformed to improve neural network and regression performance:
- **Engineered Features:** Calculated `car_age`, total `feature_count`, and created a logarithmic distribution of the price (`price_log`) to normalize right-skewed economic data.
- **Categorical Encoding:** Converted text data (Transmission, Body Type, Assembly) into Label Encoded numeric matrices.
- **EDA Visualization:** Plotted correlation heatmaps, price distributions, and categorical market shares to understand underlying trends (visualized in the Streamlit Dashboard).

## 5. Data Mining Technique / Algorithms (Open Ended Lab 02)
We applied **Random Forest Regression**, a powerful ensemble machine learning algorithm.
- **Why Random Forest?** Unlike linear regression, Random Forest is highly robust against outliers (common in pricing data) and can easily handle non-linear relationships without requiring massive amounts of hyperparameter tuning.
- The model was trained using `scikit-learn` in a cloud environment (`train_model_colab.ipynb`), allowing for fast computation over the high-dimensional matrix.

## 6. Results and Evaluation Metrics
The model was evaluated using standard regression metrics:
- **$R^2$ (Accuracy Score):** Represents the percentage of the variance in the target variable (Price) that is explained by the model. *(Check your Colab output to insert the exact % in your PPT)*
- **Mean Absolute Error (MAE):** Represents the average error margin in Pakistani Rupees. *(Check your Colab output for the exact PKR amount)*
- **Feature Importance:** The model revealed that the most influential factors dictating a car's price are the **Manufacturing Year**, **Engine CC**, **Car Model**, and **Mileage**.

## 7. Digital Dashboard
We built a real-time, interactive web application using **Streamlit** (`dashboard.py`).
The dashboard serves two main purposes:
1. **Interactive EDA:** Users can explore market distributions, pricing correlations, and dataset statistics dynamically without writing code.
2. **AI Price Estimator:** A deployed module of the trained Random Forest model (`car_price_model.pkl`). Users can input specific vehicle parameters (e.g., Toyota Corolla, 2018, 50,000 km) and receive an instant, accurate estimate of the car's fair market value.

## 8. Conclusion
The project successfully demonstrates that structured data mining pipelines, when combined with ensemble machine learning techniques, can effectively solve information asymmetry in unstructured markets. By leveraging scraped data, advanced feature engineering, and Random Forest Regression, we created a highly accurate and interactive system that empowers buyers and sellers to make data-driven pricing decisions.

---

### How to Run the Project Locally
1. **Install Requirements:** `pip install -r requirements.txt`
2. **Run the Dashboard:** `streamlit run dashboard.py`
3. **Train the Model Locally:** `python train_model.py` (Alternatively, use `train_model_colab.ipynb` on Google Colab).
