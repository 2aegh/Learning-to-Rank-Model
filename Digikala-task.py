# Importing libraries
import pandas as pd
import numpy as np

# Machine learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score, make_scorer, average_precision_score, precision_score, r2_score

# Classification and regression models
#import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

# Dimensionality reduction
from sklearn.decomposition import TruncatedSVD

#save models
import joblib

# Load datasets
query_product_data = pd.read_csv(r"C:\Users\amira\Downloads\digikala_ds_task_query_product.csv")
products_data = pd.read_csv(r"C:\Users\amira\Downloads\digikala_ds_task_products.csv")
print(query_product_data.head(2))
print(query_product_data.isnull().agg(['sum', 'mean']))
#print(query_product_data.isnull().sum())
#print(query_product_data.isnull().mean() * 100)
print('\n')
print(query_product_data.shape)
print('\n')
print(products_data.head(2))
print(products_data.isnull().agg(['sum', 'mean']))
#print(products_data.isnull().sum())
#print(products_data.isnull().mean() * 100)
print('\n')
print(products_data.shape)
# Data Preparation
# Aggregate products_data by d_id
agg_funcs = {
    'search_view': ['sum', 'mean', 'min', 'max', 'std'],
    'search_click': ['sum', 'mean', 'min', 'max', 'std'],
    'search_sales': ['sum', 'mean', 'min', 'max', 'std']
}

aggregated_products = products_data.groupby('d_id').agg(agg_funcs).reset_index()
aggregated_products.columns = ['_'.join(col) for col in aggregated_products.columns]
aggregated_products = aggregated_products.rename(columns={'d_id_': 'd_id'})
aggregated_products=aggregated_products.loc[:, ['d_id'] + [col for col in aggregated_products.columns if 'sum' in col and col != 'd_id']]
# Merge datasets on d_id
#left:291,260 inner:132,142 
data = pd.merge(query_product_data, aggregated_products, on='d_id', how='left') 
categorical_cols = ['category_id', 'brand_id']
numerical_cols = [col for col in data.columns if data[col].name not in ['brand_id', 'category_id']]
def detect_and_cap_outliers(df, column):  
    # Skip outlier detection for 'd_id' and 'q_id' columns  
    if column in ['d_id', 'q_id']:  
        print(f"Skipping outlier detection for column: {column}\n")  
        return df  

    # Calculate Q1, Q3, and IQR  
    Q1 = df[column].quantile(0.25)  
    Q3 = df[column].quantile(0.75)  
    IQR = Q3 - Q1  
    lower_bound = Q1 - 1.5 * IQR  
    upper_bound = Q3 + 1.5 * IQR  

    # Adjust lower bound if it's negative  
    if lower_bound < 0:  
        lower_bound = 0  

    # Track replacements  
    lower_replacement_count = df[df[column] < lower_bound].shape[0]  
    upper_replacement_count = df[df[column] > upper_bound].shape[0]  

    # Count total non-null entries  
    total_non_null = df[column].notnull().sum()  

    # Cap outliers  
    df.loc[df[column] < lower_bound, column] = lower_bound  
    df.loc[df[column] > upper_bound, column] = upper_bound  

    # Calculate percentages  
    lower_replacement_percentage = (lower_replacement_count / total_non_null) * 100  
    upper_replacement_percentage = (upper_replacement_count / total_non_null) * 100  

    # Print summary  
    print(f"Column: {column}")  
    print(f"Lower Bound: {lower_bound}")  
    print(f"Upper Bound: {upper_bound}")  
    print(f"Number of Outliers Capped at Lower Bound: {lower_replacement_count}")  
    print(f"Percentage of Data Capped at Lower Bound: {lower_replacement_percentage:.2f}%")  
    print(f"Number of Outliers Capped at Upper Bound: {upper_replacement_count}")  
    print(f"Percentage of Data Capped at Upper Bound: {upper_replacement_percentage:.2f}%")  
    print(f"Total Number of Outliers: {lower_replacement_count + upper_replacement_count}")  
    print(f"Percentage of Outliers: {(lower_replacement_count + upper_replacement_count) / total_non_null * 100:.2f}%\n")  

    return df  
for col in numerical_cols:  
    data = detect_and_cap_outliers(data, col)  
# First way for handeling null values (Iterative Imputer )
'''
# Initialize Iterative Imputer  
iter_imputer = IterativeImputer(max_iter=10, random_state=0)  

# Fit and transform the data  
data[numeric_cols] = iter_imputer.fit_transform(data[numeric_cols])  
'''
# Second way for handeling null values (Prediction)
def fill_missing_with_model(data, target_col):  
    # Initial simple imputation to handle other missing values  
    simple_imputer = SimpleImputer(strategy='mean')  
    data_imputed_initial = pd.DataFrame(simple_imputer.fit_transform(data), columns=data.columns)  
    
    # Separate data into those with and without the target value  
    data_non_missing = data_imputed_initial[data[target_col].notnull()]  
    
    if data_non_missing.empty:  
        print(f"No available data to train the model for column `{target_col}`; skipping this column.")  
        return data  
    
    X_train = data_non_missing.drop(columns=[target_col])  
    y_train = data_non_missing[target_col]  
    
    # Verify there is test data  
    data_missing = data_imputed_initial[data[target_col].isnull()]  
    if data_missing.empty:  
        print(f"No missing values to fill in for column `{target_col}`.")  
        return data  
    
    X_test = data_missing.drop(columns=[target_col])  
    
    # Reduce complexity of the Random Forest  
    model = RandomForestRegressor(random_state=0, n_estimators=50, max_depth=7)  
    model.fit(X_train, y_train)  

    # Predict and fill missing values  
    predicted_values = model.predict(X_test)  
    data.loc[data[target_col].isnull(), target_col] = predicted_values  
    return data  
# Impute each column separately  
for col in numerical_cols:  
    data = fill_missing_with_model(data, col)  

# Display the null in the DataFrame after imputation  
print(data.isnull().sum())  
def fill_categorical_with_model(data, target_col):  
    # Simple imputation for numeric columns, excluding the target column for now  
    numeric_cols = data.select_dtypes(include=[np.number]).columns.drop(target_col)  
    simple_imputer = SimpleImputer(strategy='mean')  
    data[numeric_cols] = simple_imputer.fit_transform(data[numeric_cols])  

    # Split into data with and without the target column missing  
    data_non_missing = data[data[target_col].notnull()]  
    data_missing = data[data[target_col].isnull()]  

    if data_non_missing.empty:  
        print(f"No available data to train the model for column `{target_col}`; skipping this column.")  
        return data  

    # Define X and y for model training  
    X_train = data_non_missing.drop(columns=[target_col])  
    y_train = data_non_missing[target_col].astype(int)  

    if data_missing.empty:  
        print(f"No missing values to fill in for column `{target_col}`.")  
        return data  

    X_test = data_missing.drop(columns=[target_col])  

    # Use a simpler and faster model: Decision Tree Classifier  
    model = DecisionTreeClassifier(random_state=0, max_depth=7)  
    model.fit(X_train, y_train)  

    # Predict and fill missing values  
    predicted_classes = model.predict(X_test)  
    data.loc[data[target_col].isnull(), target_col] = predicted_classes  
    return data  

# Impute each column separately  
for col in categorical_cols:  
    data = fill_categorical_with_model(data, col)  

# Display the null in the DataFrame after imputation  
print(data.isnull().sum())  
#data[data.isnull().any(axis=1)]
#data.query('q_id == 4219')
# Feature Engineering
data['price_discount'] = data['price'] * (1 - data['discount'] / 100)
data['click_through_rate'] = data['search_click_sum'] / data['search_view_sum']
data['conversion_rate'] = data['search_sales_sum'] / data['search_view_sum']
data.fillna(0, inplace=True)
data['rank'] = data.groupby('q_id')['target_score'].rank("dense", ascending=False)
# Encoding categorical col and Reduce the dataset's dimensionality
def encode_reduce_and_combine(data, categorical_features, non_categorical_features, n_components=5):  
    """  
    Encodes specified categorical columns, reduces their dimensionality using TruncatedSVD, and combines them with non-categorical columns.  

    Parameters:  
    - data: pd.DataFrame   
        The input DataFrame containing all the data.  
    - categorical_features: list of str  
        The list of column names to be one-hot encoded.  
    - non_categorical_features: list of str  
        The list of column names to retain without encoding.  
    - n_components: int  
        The number of components to keep from the encoded features.  

    Returns:  
    - pd.DataFrame  
        A DataFrame containing both reduced encoded features and non-categorical columns.  
    """  
    
    # One-hot encode categorical features with sparse_output  
    ohe = OneHotEncoder(drop='first', sparse_output=True)  
    
    # Create a column transformer to apply the encoder only to categorical columns  
    column_transformer = ColumnTransformer(  
        transformers=[  
            ('ohe', ohe, categorical_features)  
        ],  
        remainder='passthrough'  
    )  

    # Create a pipeline to encode and apply TruncatedSVD  
    pipeline = Pipeline(steps=[  
        ('encode', column_transformer),  
        ('reduce_dims', TruncatedSVD(n_components=n_components))  
    ])  

    # Extract non-categorical features  
    non_categorical_data = data[non_categorical_features]  

    # Fit and transform the categorical data using the pipeline  
    transformed_data_reduced = pipeline.fit_transform(data[categorical_features])  

    # Convert the reduced data to a DataFrame  
    reduced_columns = [f'svd_component_{i+1}' for i in range(n_components)]  
    reduced_df = pd.DataFrame(transformed_data_reduced, columns=reduced_columns)  

    # Concatenate the reduced data with non-categorical features  
    final_df = pd.concat([non_categorical_data,reduced_df .reset_index(drop=True)], axis=1)  

    return final_df  
try:  
    # Example usage with a DataFrame 'data' and specified feature lists  
    categorical_features = ['category_id', 'brand_id']  
    non_categorical_features = [col for col in data.columns if col not in categorical_features]  
    
    final_df = encode_reduce_and_combine(data, categorical_features, non_categorical_features, n_components=5)  
    print(final_df.head())  
except Exception as e:  
    print(f"An error occurred: {e}")  
final_df.to_csv('dk-task.csv') 
# Create Learning to Rank (LTR)
def clean_data(df):  
    # Replace inf/-inf with NaN  
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  
    
    # Optionally fill NaN with column means or medians  
    # You could use df.fillna(df.median(), inplace=True) or a similar approach  
    df.fillna(df.mean(), inplace=True)  
    return df  
# Function to scale features
def scale_features(df, features):
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df
def calculate_metrics(y_true_ranks, y_pred_ranks, k=10):
    # Ensure ranks are in the same format
    y_true_ranks = np.array(y_true_ranks)
    y_pred_ranks = np.array(y_pred_ranks)
    
    # NDCG Calculation
    ndcg = ndcg_score([y_true_ranks], [y_pred_ranks], k=k)
    
    # Mean Average Precision (MAP)
    y_true_relevance = y_true_ranks > 0  # assuming positive ranks are relevant
    map_score = average_precision_score(y_true_relevance, -y_pred_ranks)  # Use negative for descending order
    
    # Mean Reciprocal Rank (MRR)
    def reciprocal_rank(y_true_ranks):
        for i, rank in enumerate(y_true_ranks):
            if rank > 0:  # Assuming positive ranks are relevant
                return 1.0 / (i + 1)
        return 0.0
    
    mrr_score = reciprocal_rank(y_true_ranks)
    
    # Precision at k
    def precision_at_k(y_true_ranks, k):
        relevant = np.sum(y_true_ranks[:k] > 0)
        return relevant / k
    
    precision = precision_at_k(y_true_ranks, k)
    
    return ndcg, map_score, mrr_score, precision
def create_groups(q_id):
    _, q_group = np.unique(q_id, return_counts=True)
    return q_group
def ndcg_scorer(y_true, y_pred):
    return ndcg_score([y_true], [y_pred])
def hyperparameter_tuning(X_train, y_train, group_train):
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.1, 0.01],
        'max_depth': [6, 8],
        'n_estimators': [100, 200],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    best_params = {}
    best_score = -float('inf')

    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            for n_estimators in param_grid['n_estimators']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        ranker = xgb.XGBRanker(
                            objective='rank:pairwise',
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            ndcg_exp_gain=False
                        )
                        
                        # Fit model
                        ranker.fit(X_train, y_train, group=group_train)
                        
                        # Predict on training set
                        y_pred_train = ranker.predict(X_train)
                        
                        # Calculate NDCG score
                        ndcg = ndcg_score([y_train], [y_pred_train])
                        
                        # Update best parameters
                        if ndcg > best_score:
                            best_score = ndcg
                            best_params = {
                                'learning_rate': learning_rate,
                                'max_depth': max_depth,
                                'n_estimators': n_estimators,
                                'subsample': subsample,
                                'colsample_bytree': colsample_bytree
                            }
    
    return best_params
def convert_scores_to_ranks_within_groups(q_ids, y_pred):
    df = pd.DataFrame({'q_id': q_ids, 'y_pred': y_pred}).reset_index(drop=True)
    ranks = np.zeros(len(df), dtype=int)
    
    for q_id, group_indices in df.groupby('q_id').groups.items():
        group_ranks = np.argsort(np.argsort(-df.loc[group_indices, 'y_pred'])) + 1
        ranks[group_indices] = group_ranks
    
    return ranks
# Pairwise Model with Hyperparameter Tunning
df=final_df.copy()
df=clean_data(df)
features = ['relevancy_score_1', 'relevancy_score_2', 'price_discount', 'click_through_rate', 'conversion_rate', 'price', 'discount','svd_component_1','svd_component_2',
            'svd_component_3','svd_component_4','svd_component_5']
df = scale_features(df, ['relevancy_score_1', 'relevancy_score_2', 'price_discount', 'click_through_rate', 'conversion_rate', 'price', 'discount'])

X = df[features]
y = df['target_score']
q_id = df['q_id']

# Split data
X_train, X_val, y_train, y_val, q_train, q_val = train_test_split(X, y, q_id, test_size=0.2, random_state=42)

# Store ranks for validation
val_ranks = df.loc[q_val.index, 'rank'].values
group_train = create_groups(q_train)
group_val = create_groups(q_val)

# Perform hyperparameter tuning
best_params = hyperparameter_tuning(X_train, y_train, group_train)
print("Best Parameters:", best_params) #Best Parameters: {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 200, 'subsample': 0.9, 'colsample_bytree': 0.9}
# Define and fit the model with best parameters
ranker = xgb.XGBRanker(
    objective='rank:pairwise',
    ndcg_exp_gain=False,
    **best_params
)

ranker.fit(X_train, y_train, group=group_train, eval_set=[(X_val, y_val)], eval_group=[group_val], verbose=True)

# Predict on validation set
y_pred = ranker.predict(X_val)

# Convert the prediction scores to ranks within each group
y_pred_ranks = convert_scores_to_ranks_within_groups(q_val, y_pred)

# Calculate metrics
ndcg, map_score, mrr_score, precision = calculate_metrics(val_ranks, y_pred_ranks, k=10)

print(f'NDCG score: {ndcg}')
print(f'MAP score: {map_score}')
print(f'MRR score: {mrr_score}')
print(f'Precision: {precision}')
# Listwise Model
# Load and preprocess your data
df=final_df.copy()
df = clean_data(df)
features = ['relevancy_score_1', 'relevancy_score_2', 'price_discount', 'click_through_rate', 'conversion_rate', 'price', 'discount','svd_component_1','svd_component_2',
            'svd_component_3','svd_component_4','svd_component_5']
df = scale_features(df, ['relevancy_score_1', 'relevancy_score_2', 'price_discount', 'click_through_rate', 'conversion_rate', 'price', 'discount'])

X = df[features]
y = df['target_score']
q_id = df['q_id']

# Split data
X_train, X_val, y_train, y_val, q_train, q_val = train_test_split(X, y, q_id, test_size=0.2, random_state=42)

# Store ranks for validation
val_ranks = df.loc[q_val.index, 'rank'].values
group_train = create_groups(q_train)
group_val = create_groups(q_val)
# Define and fit the list-wise model
ranker2 = xgb.XGBRanker(objective='rank:ndcg',ndcg_exp_gain=False)
ranker2.fit(X_train, y_train, group=group_train, eval_set=[(X_val, y_val)], eval_group=[group_val], verbose=True)

# Predict on validation set
y_pred = ranker2.predict(X_val)

# Convert the prediction scores to ranks within each group
y_pred_ranks = convert_scores_to_ranks_within_groups(q_val, y_pred)

# Calculate metrics
ndcg, map_score, mrr_score, precision = calculate_metrics(val_ranks, y_pred_ranks, k=10)

print(f'NDCG score: {ndcg}')
print(f'MAP score: {map_score}')
print(f'MRR score: {mrr_score}')
print(f'Precision: {precision}')
# Convert to numpy arrays if not already
val_ranks = np.array(val_ranks)
y_pred_ranks = np.array(y_pred_ranks)

# Count matches
matches = np.sum(val_ranks == y_pred_ranks)
total_items = len(val_ranks)

# Print the counts
print(f'Number of matching ranks: {matches}')
print(f'Total number of items: {total_items}')
print(f'Percentage of matches: {matches / total_items * 100:.2f}%')
# Pointwise Method With Optimized Randomforest
def optimize_rf_with_internal_sampling(X_train, y_train, X_test, y_test, sample_size=0.1, cv_folds=3, n_iter_search=20):  
    # Create a smaller subset of the training data  
    X_train_sample, _, y_train_sample, _ = train_test_split(  
        X_train, y_train, train_size=sample_size, random_state=42  
    )  

    # Define the model  
    pointwise_model = RandomForestRegressor(random_state=42)  

    # Define parameter grid for RandomizedSearchCV  
    param_distributions = {  
        'n_estimators': [50, 100, 200],  
        'max_depth': [None, 10, 20],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4]  
    }  

    # Perform Randomized Search with cross-validation on sample  
    random_search = RandomizedSearchCV(  
        estimator=pointwise_model,  
        param_distributions=param_distributions,  
        n_iter=n_iter_search,  
        scoring='r2',  
        refit=True,  
        cv=cv_folds,  
        verbose=1,  
        n_jobs=-1,  
        random_state=42  
    )  

    # Fit the model on the sample subset  
    random_search.fit(X_train_sample, y_train_sample)  

    # Get the best parameters from the search  
    best_params = random_search.best_params_  
    print("Best Parameters from Sample Subset:", best_params)  

    # Train the model with best parameters on the full training set  
    best_model_full = RandomForestRegressor(**best_params, random_state=42)  
    best_model_full.fit(X_train, y_train)  

    # Evaluate R-squared on training set  
    y_train_pred = best_model_full.predict(X_train)  
    r_squared_train = r2_score(y_train, y_train_pred)  

    # Evaluate R-squared on test set  
    y_test_pred = best_model_full.predict(X_test)  
    r_squared_test = r2_score(y_test, y_test_pred)  

    return best_model_full, best_params, r_squared_train, r_squared_test  
best_model, best_params, r_squared_train, r_squared_test = optimize_rf_with_internal_sampling(  
    X_train, y_train, X_val, y_val, sample_size=0.1  
)  

# Output the results  
# Best Parameters from Sample Subset: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 10}

print("Optimized Random Forest Regressor")  
print("Best Hyperparameters for Full Data:", best_params)  
print("R-squared on Full Training Set:", r_squared_train)  
print("R-squared on Test Set:", r_squared_test)  
# Saving the models
# Define file names for saving the models
pairwise_model_filename = 'pairwise_model.pkl'
listwise_model_filename = 'listwise_model.pkl'
pointwise_model_filename = 'pointwise_model.pkl'

# Save models using joblib
joblib.dump(ranker, pairwise_model_filename)
joblib.dump(ranker2, listwise_model_filename)
joblib.dump(best_model, pointwise_model_filename)

print(f"Models saved as {pairwise_model_filename}, {listwise_model_filename}, and {pointwise_model_filename}.")

# Drafts
'''
# Function to scale either specific or all numeric columns  
def scale_numeric_columns(df, method='standard', columns=None):  
    # Clean the data before scaling  
    df_cleaned = clean_data(df)  
    
    # Determine which columns to scale  
    if columns is None:  
        columns = df_cleaned.select_dtypes(include=[np.number]).columns  
    
    if method == 'standard':  
        scaler = StandardScaler()  
    elif method == 'minmax':  
        scaler = MinMaxScaler()  
    else:  
        raise ValueError("Method must be either 'standard' or 'minmax'.")  
    
    df_scaled = df_cleaned.copy()  
    df_scaled[columns] = scaler.fit_transform(df_cleaned[columns])  
    
    return df_scaled  
    
# Combine actual and predicted ranks for inspection
results_df = pd.DataFrame({
    'True Rank': val_ranks,
    'Predicted Rank': y_pred_ranks
})

# Display a sample of the results
print(results_df.head(20))  # Adjust the number as needed

# Inverse transformation to approximate original one-hot encoded values  
approx_reconstructed_data = pca.inverse_transform(one_hot_pca)  

# Scale back the data to its original scale  
approx_reconstructed_data = scaler.inverse_transform(approx_reconstructed_data)  

# Convert back to a DataFrame with original one-hot column names  
approx_reconstructed_df = pd.DataFrame(approx_reconstructed_data, columns=one_hot_columns)  
print("Approximated Reconstructed One-Hot Encoded Data:")  
print(approx_reconstructed_df.head()) 


# Define your categorical columns explicitly.  
categorical_cols = ['category_id', 'brand_id']  

# Automatically determine other numerical columns excluding your specific categorical columns  
numeric_cols = data.columns.difference(categorical_cols).tolist()  

# Define numerical transformer
 
numeric_transformer = Pipeline(steps=[  
    ('imputer', KNNImputer(n_neighbors=5)),  
    ('scaler', StandardScaler())  
])  

numeric_transformer = Pipeline(steps=[  
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler())  
])  
# Define categorical transformer  
categorical_transformer = Pipeline(steps=[  
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  
])  

# Define ColumnTransformer to process each column group using the appropriate transformer  
preprocessor = ColumnTransformer(  
    transformers=[  
        ('num', numeric_transformer, numeric_cols),  
        ('cat', categorical_transformer, categorical_cols)  
    ]  
)  
# Fit and transform the data  
transformed_data = preprocessor.fit_transform(data)  

# Extract categorical column names after transformation  
cat_columns = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)  
'''
