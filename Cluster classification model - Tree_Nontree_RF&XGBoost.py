#Created by Jeff Anderson - Don't hesitate to email me with any questions - janderson@iegconsulting.com

import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.metrics import classification_report

# Define file paths
MSOrtho = "Filepath/Multispectral_imagery.tif" 
clusters_updated = "Filepath/updated_clusters.shp"
clusters_predicted = "Filepath/predicted_clusters.shp"

# Load the shapefile
gdf = gpd.read_file(clusters_updated)

# Define features and label
features_columns =  [
    'R_mean', 'G_mean', 'B_mean', 'N_mean', 'RE_mean', 'NDVI_mean', 'SAVI_mean', 'MSAVI_mean', 'GRNVI_mean', 'GNDVI_mean',
    'R_med', 'G_med', 'B_med', 'N_med', 'RE_med', 'NDVI_med', 'SAVI_med', 'MSAVI_med', 'GRNVI_med', 'GNDVI_med',
    'R_75', 'G_75', 'B_75', 'N_75', 'RE_75', 'NDVI_75', 'SAVI_75', 'MSAVI_75', 'GRNVI_75', 'GNDVI_75',
    'R_90', 'G_90', 'B_90', 'N_90', 'RE_90', 'NDVI_90', 'SAVI_90', 'MSAVI_90', 'GRNVI_90', 'GNDVI_90',
    'R_max', 'G_max', 'B_max', 'N_max', 'RE_max', 'NDVI_max', 'SAVI_max', 'MSAVI_max','GRNVI_max', 'GNDVI_max', 'Area', 'Perimeter', 'Aspect_Rat', 'Compactnes'
]
label_column = 'training'

# Filter data and prepare datasets
filtered_data = gdf[gdf[label_column].isin(['tree', 'non tree'])]
test_set = gdf[gdf[label_column].isin(['NULL'])]

# Training dataset
training_data = filtered_data.dropna(subset=[label_column])
X_train = training_data[features_columns]
y_train = training_data[label_column]

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Test dataset
test_data = test_set.dropna(subset=[label_column])
X_test = test_data[features_columns]

# Initialize classifiers
clf = RandomForestClassifier(random_state=0, class_weight='balanced')
xgb_clf = GradientBoostingClassifier(random_state=0)
pipeline = IMBPipeline([('oversample', SMOTE()), ('classifier', xgb_clf)])

# Define hyperparameter grids
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    # ... other hyperparameters for RandomForest
}

param_grid_xgb = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    # ... other hyperparameters for GradientBoosting
}

# Grid search with cross-validation
grid_search_rf = GridSearchCV(clf, param_grid_rf, cv=StratifiedKFold(10), scoring='accuracy')
grid_search_xgb = GridSearchCV(pipeline, param_grid_xgb, cv=StratifiedKFold(10), scoring='accuracy')

# Perform hyperparameter tuning and feature selection
grid_search_rf.fit(X_train, y_train_encoded)
grid_search_xgb.fit(X_train, y_train_encoded)

# Use best parameters to train your final models
best_rf = grid_search_rf.best_estimator_
best_xgb = grid_search_xgb.best_estimator_

# Feature selection for RandomForest
rfecv_rf = RFECV(estimator=best_rf, step=1, cv=StratifiedKFold(10), scoring='accuracy').fit(X_train, y_train_encoded)
selected_features_rf = [features_columns[i] for i in range(len(features_columns)) if rfecv_rf.support_[i]]

# Define custom importance getter function
def importance_getter(estimator):
    return estimator.named_steps['classifier'].feature_importances_

# Feature selection for XGBoost
rfecv_xgb = RFECV(estimator=best_xgb, step=1, cv=StratifiedKFold(10), scoring='accuracy', importance_getter=importance_getter).fit(X_train, y_train_encoded)
selected_features_xgb = [features_columns[i] for i in range(len(features_columns)) if rfecv_xgb.support_[i]]

# Train models on selected features
X_train_rf_selected = X_train[selected_features_rf]
X_train_xgb_selected = X_train[selected_features_xgb]
best_rf.fit(X_train_rf_selected, y_train_encoded)
best_xgb.fit(X_train_xgb_selected, y_train_encoded)

# Apply feature selection to test data
X_test_rf_selected = X_test[selected_features_rf]
X_test_xgb_selected = X_test[selected_features_xgb]

# Apply feature selection to the split data
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_rf_selected, y_train_encoded, test_size=0.2, random_state=0
)

# Train on the split training set with selected features
best_rf.fit(X_train_split, y_train_split)
best_xgb.fit(X_train_split, y_train_split)

# Predict on the validation set
y_val_pred_rf = best_rf.predict(X_val_split)
y_val_pred_xgb = best_xgb.predict(X_val_split)

# Compute the confusion matrix
cm_rf = confusion_matrix(y_val_split, y_val_pred_rf)
cm_xgb = confusion_matrix(y_val_split, y_val_pred_xgb)

# Get unique class labels from the training set (transformed back to original labels)
class_labels = label_encoder.classes_

# Create a DataFrame from the confusion matrix
cm_rf_df = pd.DataFrame(cm_rf, index=class_labels, columns=class_labels)
cm_xgb_df = pd.DataFrame(cm_xgb, index=class_labels, columns=class_labels)

# Print the confusion matrices
print("RF Confusion Matrix:")
print(cm_rf_df)
print("\nXGB Confusion Matrix:")
print(cm_xgb_df)


# Predict probabilities with RandomForest
y_test_proba_rf = best_rf.predict_proba(X_test_rf_selected)

# Ensure X_test has the same columns in the same order as X_train_xgb_selected
X_test_aligned = X_test[X_train_split.columns]

# Debugging: Print column names to check alignment
print("Columns in X_train_xgb_selected:", X_train_split.columns)
print("Columns in X_test_aligned:", X_test_aligned.columns)

# Predict probabilities with XGBoost
y_test_proba_xgb = best_xgb.predict_proba(X_test_aligned)

# Extract maximum probability as confidence score
test_data['rf_confidence'] = y_test_proba_rf.max(axis=1)
test_data['xgb_confidence'] = y_test_proba_xgb.max(axis=1)

# Correct extraction of the predicted class
test_data['rf_pred'] = label_encoder.inverse_transform(np.argmax(y_test_proba_rf, axis=1))
test_data['xgb_pred'] = label_encoder.inverse_transform(np.argmax(y_test_proba_xgb, axis=1))

# Print selected features
print("Selected features for RandomForest:", selected_features_rf)
print("Selected features for XGBoost:", selected_features_xgb)

# Combine the predicted and training data
gdf_combined = pd.concat([training_data, test_data])

# Now save the updated GeoDataFrame
gdf_combined.to_file(clusters_predicted)

# Classification report for RandomForest
print("\nRandomForest Classification Report:")
print(classification_report(y_val_split, y_val_pred_rf, target_names=class_labels))

# Classification report for XGBoost
print("\nXGBoost Classification Report:")
print(classification_report(y_val_split, y_val_pred_xgb, target_names=class_labels))

# End Script
