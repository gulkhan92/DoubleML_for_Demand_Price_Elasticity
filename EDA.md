# Demand-Price Dynamics: Exploratory Data Analysis

This analysis characterizes the high-dimensional feature space of the M5 Walmart dataset to validate the assumptions required for Causal Inference. The primary objective is to isolate the relationship between price shifts and quantity demanded while identifying the confounding variables that the Double Machine Learning (DML) framework must "partial out."

---
## 1. Objectives of Analysis
1. **Characterize Signal Strength**: Evaluate the frequency and magnitude of price changes (Treatment Variation).
2. **Diagnose Endogeneity**: Identify correlations between price and historical demand (Lagged Sales).
3. **Identify Confounding Factors**: Map the influence of external shocks (SNAP events, holidays) on sales volume.

## 2. Statistical Data Characteristics

The dataset, after initial loading and feature engineering, has the following structure:

```
<class 'pandas.DataFrame'>
Index: 11303955 entries, 2 to 11332694
Data columns (total 19 columns):
 #   Column        Dtype  
---  ------        -----  
 0   item_id       str    
 1   store_id      int8   
 2   wm_yr_wk      int64  
 3   dept_id       int8   
 4   state_id      int8   
 5   month         int64  
 6   year          int64  
 7   snap_CA       int64  
 8   snap_TX       int64  
 9   snap_WI       int64  
 10  event_name_1  int8   
 11  event_type_1  int8   
 12  sales         int64  
 13  date          str    
 14  sell_price    float64
 15  log_quantity  float64
 16  log_price     float64
 17  lag_sales_1   float64
 18  lag_sales_2   float64
dtypes: float64(5), int64(7), int8(5), str(2)
memory usage: 1.3 GB
```

### Descriptive Statistics and Sample Volume

```
       log_quantity     log_price         sales    sell_price
count  1.130396e+07  1.130396e+07  1.130396e+07  1.130396e+07
mean   8.930676e-01  9.984020e-01  4.051355e+00  3.248750e+00
std    1.013140e+00  6.138056e-01  1.346255e+01  2.130838e+00
min    0.000000e+00 -4.605170e+00  0.000000e+00  1.000000e-02
25%    0.000000e+00  6.830968e-01  0.000000e+00  1.980000e+00
50%    6.931472e-01  9.858168e-01  1.000000e+00  2.680000e+00
75%    1.609438e+00  1.378766e+00  4.000000e+00  3.970000e+00
max    7.976595e+00  2.969388e+00  2.911000e+03  1.948000e+01
```

**Observations:**
*   **Sales Distribution**: The `sales` column has a minimum of 0 and a maximum of 2911, with a mean of ~4. This indicates a highly skewed distribution with many zero-sales instances and a long tail of high-volume sales. The `log_quantity` (log-transformed sales) effectively handles this skewness.
*   **Price Range**: `sell_price` ranges from $0.01 to $19.48. The `log_price` transformation normalizes this range.

### Distribution of Sales Volume (Plot: `sales_distribution.png`)

The histogram of `sales` (with a log-scaled y-axis) visually confirms the highly skewed nature of sales data, with a large concentration at lower sales volumes and a long tail extending to very high sales. This justifies the use of `log1p` transformation for the outcome variable (`log_quantity`) to normalize its distribution and make it more suitable for regression models.

### Total Weekly Sales Over Time (Plot: `weekly_sales_trend.png`)

This plot reveals the overall temporal patterns in demand. We expect to see:
*   **Seasonality**: Clear weekly, monthly, or yearly cycles.
*   **Trends**: Potential upward or downward trends over the observed period.
*   **Event Impacts**: Visible spikes or dips corresponding to major holidays or events (e.g., Thanksgiving, Christmas, SNAP disbursement weeks).

These patterns highlight the importance of including time-based features and event flags as controls (`W`) in the DML model to account for these confounding factors.

---

## 3. Project-Specific EDA: Demand Price Elasticity

This section focuses on insights directly relevant to estimating the causal effect of price on demand.

### Log-Price vs Log-Quantity Scatter Plot (Plot: `price_quantity_correlation.png`)
The scatter plot reveals a dense "cloud" of data points with a visible, though noisy, negative slope.
*   **Negative Correlation**: The red regression line trends downward, suggesting that the "Law of Demand" holds generally across the dataset.
*   **Vertical Clustering**: Large vertical clusters at specific price points indicate that many items share common price tiers, but their sales vary wildly due to non-price factors (confounders).
*   **Non-Linearity**: The dispersion suggests that a simple linear fit (OLS) will struggle to capture the true effect, as it will be pulled by the variance in demand that price cannot explain.

### Correlation Heatmap (Plot: `correlation_heatmap.png`)

This heatmap displays the pairwise correlations between the treatment (`log_price`), outcome (`log_quantity`), and key confounders (`month`, `snap_CA`, `snap_TX`, `snap_WI`, `lag_sales_1`, `lag_sales_2`).

**Observations:**
*   **Confounder Identification**: High correlations between confounders (e.g., `month`, `snap_CA`) and both `log_price` and `log_quantity` would strongly suggest the presence of omitted variable bias in naive OLS. For instance, if `snap_CA` is positively correlated with `log_quantity` (more sales during SNAP weeks) and negatively correlated with `log_price` (Walmart offers discounts during SNAP weeks), it acts as a confounder.
*   **Demand Momentum**: There is a strong positive correlation (approx. 0.6) between `log_quantity` and its lags (`lag_sales_1`, `lag_sales_2`). This suggests that last week's sales are one of the strongest predictors of this week's sales, and failing to control for this "momentum" would lead to biased elasticity estimates.
*   **Price-Lag Relationship**: A non-negligible correlation between `log_price` and `lag_sales_1` suggests that current prices are potentially adjusted based on past performance—a clear sign of endogeneity.
*   **Multicollinearity**: High correlations among control variables themselves might indicate multicollinearity, which tree-based models can handle better than linear models.

---

## 4. Causal Diagnostics: Elasticity and Confounding

### Price Volatility (Coefficient of Variation)
A fundamental requirement for elasticity estimation is the presence of price variation. Items with "sticky prices" (zero change) provide no signal for the treatment effect.

*   **Mean Price CV**: 0.0401
*   **Median Price CV**: 0.0369
*   **Sticky Prices**: 1,343 out of 14,370 item-store pairs exhibit zero price variation.
*   **Plot Analysis (`price_cv_distribution.png`)**: The histogram of Price CV peaks sharply around 0.03-0.05. This tells us that most products experience price changes of roughly 3-5% relative to their mean. This provides a healthy "experimental signal" for DML to learn from, as there is sufficient variation to distinguish price effects from noise.

### Endogeneity Check (Plot: `price_lag_sales_correlation.png`)
This plot shows a slight upward-sloping regression line between `lag_sales_1` and `log_price`.
*   **Observation**: Higher sales in the previous week appear to be followed by slightly higher prices in the current week.
*   **Suggestion**: This suggests a "reactive pricing" strategy where Walmart may be capturing consumer surplus on high-demand items. This relationship is the definition of endogeneity: the "Treatment" (Price) is determined by the "Outcome" history, which creates a loop that standard OLS cannot break.

### Departmental Distribution (Plot: `log_quantity_by_department.png`)
The boxplot shows significant variation in `log_quantity` medians and interquartile ranges across departments (IDs 0 through 6).
*   **High-Volume Depts**: Some departments show consistently higher sales and tighter distributions.
*   **Heterogeneity Signal**: The difference in baseline demand levels suggests that shoppers' price sensitivity likely differs across these categories (e.g., staples vs. discretionary items). This justifies the use of $X$ (effect modifiers) in DML to estimate segment-specific elasticities.

---

## 5. Conclusion of EDA

The EDA confirms the complex nature of the M5 retail dataset, characterized by large scale, skewed sales distributions, strong temporal patterns, and clear indications of confounding. These observations strongly justify the choice of Double Machine Learning, which is designed to handle such high-dimensional data and complex relationships to isolate the true causal effect of price on demand. The EDA also provides initial visual and statistical cues that will be further explored and validated by the various robustness and falsification tests.