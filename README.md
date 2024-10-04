

# Learning to Rank Model for E-Commerce Search Optimization

## Introduction

This project aims to develop a robust **Learning to Rank (LTR)** model to significantly improve the relevance of search results on an e-commerce platform. The model optimizes how search results are ranked, ensuring they align closely with user intent and deliver the most relevant products or information in response to user queries.

Despite the complexity and depth of this task, the entire project was completed within a single day due to time constraints. This required intense focus and efficiency, ensuring that each stage—from data preprocessing to model implementation—was handled swiftly without compromising the quality of the final model.

## Data Description

### Dataset 1: Search Query and Product Attributes

- **File Name**: `digikala_ds_task_query_product.csv`
- **Contents**: Consists of search queries, associated product attributes, and relevance scores. These scores reflect how well a product matches a particular query and act as the target variable for the LTR model.
- **Null Values**:
  - `relevancy_score_2`: 54,770 (18.8%)
  - `category_id`, `brand_id`, `discount`: 6 (0%)
  - `price`: 167,501 (57%)

### Dataset 2: Product User Behavior

- **File Name**: `digikala_ds_task_products.csv`
- **Contents**: Includes user interaction data with products, such as the number of views, clicks, and sales. It provides insights into how users interact with products over a specific timeframe.

## Data Preprocessing

### Handling Many-to-Many Relationships

When merging the product user behavior data with the search query dataset, aggregation was performed on the product data by product ID. Statistical measures such as sum, mean, minimum, maximum, and standard deviation were calculated for each product's views, clicks, and sales. This ensured that each product ID had a single, consolidated entry for meaningful analysis.

### Outlier Detection and Handling

Outliers were detected and capped using the Interquartile Range (IQR) method for numerical columns, excluding identifiers like product and query IDs. Negative lower bounds were adjusted to zero to maintain logical consistency.

### Handling Null Values

#### Integer Missing Value Imputation

1. **Initial Imputation**: Applied simple imputation strategies to ensure the dataset was sufficiently complete.
2. **Predictive Modeling**: Employed a Random Forest Regressor to predict and fill in missing integer values by:
   - Dividing the dataset into parts with and without missing values.
   - Training the model on non-missing data.
   - Using the model to estimate and fill missing values, leveraging data relationships for accuracy.

#### Categorical Missing Value Imputation

1. **Most Frequent Category Imputation**: Replaced missing categorical values with the most frequently occurring category.
2. **Predictive Modeling**: Utilized a Decision Tree Classifier to predict and fill in missing categorical values, capturing non-linear patterns for improved accuracy.

## Feature Engineering

- **Price Discount**: Calculated the effective price after applying discounts for a more accurate price representation.
- **Click-Through Rate (CTR)**: Computed by dividing the number of clicks by the number of views to measure user engagement.
- **Conversion Rate**: Determined by dividing the number of sales by the number of views to assess the efficiency of converting views into sales.
- **Ranking within Queries**: Created a dense rank feature within each query group based on target scores to understand relative importance.
- **Missing Value Imputation for New Features**: Filled any resulting null values with zero to ensure continuity.

## Handling Categorical Features

- **One-Hot Encoding**: Transformed categorical columns (`category_id` and `brand_id`) using one-hot encoding, expanding the dimensionality significantly.
- **Dimensionality Reduction**: Applied Singular Value Decomposition (SVD) to reduce the high-dimensional encoded columns into 5 components, managing memory and computation efficiently.
- **Sparse Representation**: Utilized sparse matrices to handle the large number of dimensions effectively, addressing memory usage and processing speed challenges.

## Model Building and Tuning

### Pairwise Ranking Model

- **Algorithm**: `XGBRanker` for pairwise ranking.
- **Parameter Tuning**: Conducted grid search to optimize hyperparameters.
- **Best Parameters**:
  - `learning_rate`: 0.1
  - `max_depth`: 8
  - `n_estimators`: 200
  - `subsample`: 0.9
  - `colsample_bytree`: 0.9

### Listwise Ranking Model

- **Algorithm**: `XGBRanker` focusing on optimizing the entire list for ranking metrics like NDCG.

### Pointwise Model with Random Forest Regressor

- **Algorithm**: `Random Forest Regressor` to predict relevance scores independently.
- **Parameter Tuning**: Used a sample subset for efficient hyperparameter exploration.
- **Best Parameters**:
  - `n_estimators`: 200
  - `max_depth`: 10
  - `min_samples_split`: 10
  - `min_samples_leaf`: 1

## Results

### Pairwise Ranking Model

- **NDCG Score**: 0.4615

### Listwise Ranking Model

- **NDCG Score**: 0.5253

### Pointwise Model with Random Forest Regressor

- **R-squared on Full Training Set**: 0.2720
- **R-squared on Test Set**: 0.2379

## Discussion

- **Anomalies in Evaluation Metrics**: The perfect scores in MAP, MRR, and Precision suggest potential issues with data handling, model setup, or evaluation processes. Further investigation is needed to ensure metrics accurately reflect the model's performance.
- **Model Comparison**: The Listwise model outperformed the Pairwise model in NDCG, indicating more effective ranking at the list level.
- **Predictive Power**: The Random Forest Regressor's modest R-squared values indicate room for improving the model's predictive capabilities.

### Observations and Future Considerations

- **Data Merging Strategy**: Using a left join resulted in many null entries. An inner join could reduce null values and potentially improve model outcomes.
- **Time Constraints**: Due to limited time, a comprehensive revision wasn't feasible. Future work should explore different merging strategies to enhance data alignment and reduce noise.
- **Metric Evaluation**: Re-evaluate data integrity, preprocessing steps, and model configurations to ensure realistic assessments of model performance.

## Contact
For questions or feedback, please contact **Amirali Eghtesad** at amirali.egh@gmail.com .

---

