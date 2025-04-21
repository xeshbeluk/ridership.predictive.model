import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, accuracy_score, roc_curve

from sklearn.model_selection import KFold
import statsmodels.formula.api as smf

 # Load and clean data
ridership_data_raw = pd.read_csv("combined_data_no_prop.csv")
stops_centrality   = pd.read_csv("stops_routes_w_centrality.csv")
land_use           = pd.read_csv("stops_land_use.csv")

ridership_data = ridership_data_raw.copy()
ridership_data['total_riders'] = ridership_data['TOTAL_ONS'] + ridership_data['TOTAL_OFFS']
ridership_data = ridership_data.replace(0, 1)

ids_complete = (
    ridership_data.groupby('LOCATION_ID')['month']
    .nunique()
    .pipe(lambda s: s[s == 12].index)
)
ridership_data = ridership_data[ridership_data['LOCATION_ID'].isin(ids_complete)].dropna()

cols_to_drop = [
    'LOCATION_ID', 'stop_name', 'route_short_name', 'direction', 'month',
    'stop_sequence', 'stop_id', 'stop_code', 'total_riders'
]
training = (
    ridership_data
    .drop(columns=[c for c in cols_to_drop if c in ridership_data.columns])
    .drop_duplicates(subset='LOCATION_ID')
)

# Log‑transform and z‑score selected continuous variables
continuous_cols = training.columns  
training[continuous_cols] = (
    np.log2(training[continuous_cols])
    .pipe(lambda df: pd.DataFrame(StandardScaler().fit_transform(df),
                                  columns=df.columns,
                                  index=df.index))
)

# PCA
pca = PCA()
pc_scores = pca.fit_transform(training)

explained = pca.explained_variance_ratio_.cumsum()
print("Cumulative variance explained by PCs:")
print(explained)

# Scree plot
plt.figure()
plt.plot(range(1, len(pca.explained_variance_)+1), pca.explained_variance_, marker='o')
plt.axhline(y=1, linestyle='--')
plt.title("Scree Plot")
plt.xlabel("PC")
plt.ylabel("Eigenvalue")
plt.tight_layout()
plt.show()

# Silhouette oprimal number of clusters, we use the first 8 PCs

X_pca8 = pc_scores[:, :8]    
silhouette_avgs = {}

for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=123)
    labels = km.fit_predict(X_pca8)
    silhouette_avgs[k] = silhouette_score(X_pca8, labels)

best_k = max(silhouette_avgs, key=silhouette_avgs.get)
print(f"Best k by average silhouette width = {best_k}")

plt.figure()
plt.plot(list(silhouette_avgs.keys()),
         list(silhouette_avgs.values()),
         marker='o')
plt.title("Silhouette Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Average silhouette width")
plt.tight_layout()
plt.show()

# K-means clustering 

km_final = KMeans(n_clusters=best_k, n_init=10, random_state=123)
cluster_labels = km_final.fit_predict(X_pca8)

cluster_df = (
    pd.DataFrame({'LOCATION_ID': training.index, 'cluster': cluster_labels})
    .set_index('LOCATION_ID')
)

ridership_clusters = (
    ridership_data
    .merge(cluster_df, left_on='LOCATION_ID', right_index=True, how='left')
)

stops_networks_clusters = (
    ridership_clusters
    .merge(
        stops_centrality[['name', 'degree', 'betweenness', 'closeness']],
        left_on='LOCATION_ID', right_on='name', how='left'
    )
)

ridership_combined = (
    stops_networks_clusters
    .merge(land_use, on='LOCATION_ID', how='left')
    .drop(columns=['X', 'Y'], errors='ignore')
)

# Adding land use a category, then label
if 'ZONEGEN_CL' in ridership_combined.columns:
    ridership_combined['ZONEGEN_CL'] = ridership_combined['ZONEGEN_CL'].astype('category')

ridership_combined['low_ridership'] = (
    ridership_combined
    .groupby('cluster')['total_riders']
    .transform(lambda x: (x < x.quantile(0.25)).astype(int))
).astype(int)

# Standardizing predictors
numeric_predictors = [
    'estimated_pop_density', 'white_nh', 'black', 'native', 'asian', 'pacific',
    'hispanic', 'other_race', 'two_races', 'male',
    'age_below_18', 'age_18_29', 'age_30_44', 'age_45_54', 'age_55_64',
    'educ_no_school', 'educ_master', 'educ_prof', 'educ_phd',
    'english_native', 'average_household_size', 'median_income',
    'degree', 'betweenness', 'closeness'
]
existing_predictors = [c for c in numeric_predictors if c in ridership_combined.columns]
ridership_combined[existing_predictors] = StandardScaler().fit_transform(
    ridership_combined[existing_predictors]
)

# Drop rows with NA in predictors or outcome
df = ridership_combined.dropna(subset=existing_predictors + ['low_ridership', 'cluster', 'month'])

# Convert categoricals
df['cluster'] = df['cluster'].astype('category')
df['month']   = df['month'].astype('category')
if 'ZONEGEN_CL' in df.columns:
    df['ZONEGEN_CL'] = df['ZONEGEN_CL'].astype('category')

# GLMM 

formula = (
    'low_ridership ~ '
    ' + '.join(existing_predictors) +
    ' + C(cluster) + C(month)' + # We approximate the random intercepts by including them as categorical fixed effects 
    (' + C(ZONEGEN_CL)' if 'ZONEGEN_CL' in df.columns else '')  
)

logit_model = smf.logit(formula=formula, data=df).fit(maxiter=100, disp=True)
print(logit_model.summary())

# Post-hoc diag
df['pred_prob'] = logit_model.predict(df)
auc_full = roc_auc_score(df['low_ridership'], df['pred_prob'])
print(f"Full‑sample AUC = {auc_full:.3f}")

fpr, tpr, _ = roc_curve(df['low_ridership'], df['pred_prob'])
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc_full:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# Residuals
df['resid'] = logit_model.resid_response
outliers = df[np.abs(df['resid']) > 2]
print(f"Identified {outliers.shape[0]} potential outliers (|resid| > 2)")

# K-fold validation
unique_stops = df['LOCATION_ID'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=123)

cv_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(unique_stops), start=1):
    train_ids = unique_stops[train_idx]
    test_ids  = unique_stops[test_idx]

    train_df = df[df['LOCATION_ID'].isin(train_ids)]
    test_df  = df[df['LOCATION_ID'].isin(test_ids)]

    model_cv = smf.logit(formula=formula, data=train_df).fit(maxiter=100, disp=False)
    test_df = test_df.copy()
    test_df['pred_prob'] = model_cv.predict(test_df)
    auc_val = roc_auc_score(test_df['low_ridership'], test_df['pred_prob'])
    acc_val = accuracy_score(test_df['low_ridership'], (test_df['pred_prob'] > 0.5).astype(int))
    cv_results.append({'fold': fold, 'auc': auc_val, 'acc': acc_val})
    print(f"Fold {fold}: AUC={auc_val:.3f}, ACC={acc_val:.3f}")

mean_auc = np.mean([x['auc'] for x in cv_results])
mean_acc = np.mean([x['acc'] for x in cv_results])
print(f"Mean CV AUC = {mean_auc:.3f}")
print(f"Mean CV Accuracy = {mean_acc:.3f}")
