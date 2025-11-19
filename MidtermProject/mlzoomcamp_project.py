#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

df = pd.read_csv("exoplanet_ml_project/exotrain.csv")  # direct path
df.head()



# In[6]:


import matplotlib.pyplot as plt

# Get index of first planet example
planet_idx = df[df['LABEL'] == 2].index[0]
row = df.iloc[planet_idx]

# Extract flux values (skip LABEL column)
flux_planet = row.values[1:]  

plt.figure(figsize=(10, 4))
plt.plot(flux_planet)
plt.title(f"Confirmed Exoplanet – LABEL=2 (Index={planet_idx})")
plt.xlabel("Time Index")
plt.ylabel("Flux")
plt.show()


# In[7]:


# Find an example of LABEL=1 (no exoplanet)
non_planet_idx = df[df['LABEL'] == 1].index[0]
row2 = df.iloc[non_planet_idx]
flux_no_planet = row2.values[1:]

plt.figure(figsize=(10,4))
plt.plot(flux_no_planet)
plt.title(f"No Exoplanet – LABEL = 1 (Index = {non_planet_idx})")
plt.xlabel("Time Index")
plt.ylabel("Flux")
plt.show()


# In[8]:


df_planet = df[df['LABEL'] == 2]                # 37 samples
df_no_planet = df[df['LABEL'] == 1].sample(len(df_planet), random_state=42)

df_balanced = pd.concat([df_planet, df_no_planet]).sample(frac=1, random_state=42)
df_balanced['LABEL'].value_counts()


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Split data
X = df_balanced.drop("LABEL", axis=1)
y = df_balanced["LABEL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train ML model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[10]:


import numpy as np

def smooth_flux(row, window_size=5):
    flux = row.values[1:]  # skip LABEL
    smoothed = np.convolve(flux, np.ones(window_size)/window_size, mode='same')
    return np.insert(smoothed, 0, row['LABEL'])  # keep LABEL at index 0


# In[11]:


df_smooth = df_balanced.apply(lambda row: smooth_flux(row), axis=1, result_type='expand')
df_smooth.head()


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Separate X and y
X = df_smooth.drop(columns=[0])      # column 0 = LABEL
y = df_smooth[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[16]:


from scipy.signal import savgol_filter
import numpy as np

def sg_smooth(row, window=25, poly=2):
    flux = row.values[1:]
    smoothed = savgol_filter(flux, window_length=window, polyorder=poly)
    return np.insert(smoothed, 0, row['LABEL'])


# In[17]:


df_smooth2 = df_balanced.apply(lambda row: sg_smooth(row), axis=1, result_type='expand')


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Separate X and y
X = df_smooth2.drop(columns=[0])      # column 0 = LABEL
y = df_smooth2[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[19]:


import matplotlib.pyplot as plt
import numpy as np

wrong_indices = X_test.index[y_test != y_pred]

for i in wrong_indices:
    row = df.iloc[i]
    flux = row.values[1:]
    plt.figure(figsize=(10,4))
    plt.plot(flux)
    plt.title(f"Misclassified Example – True = {y_test[i]}, Pred = {y_pred[i]}")
    plt.xlabel("Time Index")
    plt.ylabel("Flux")
    plt.show()


# In[20]:


wrong_indices = X_test.index[y_test != y_pred]  # original indexes

for i in wrong_indices:
    row = df.iloc[i]             # Use original dataset index
    flux = row.values[1:]        # Drop label
    plt.figure(figsize=(10,4))
    plt.plot(flux)
    plt.title(f"Misclassified Example – True = {y_test.loc[i]}, Pred = {y_pred.loc[i]}")
    plt.xlabel("Time Index")
    plt.ylabel("Flux")
    plt.show()


# In[21]:


wrong_indices = X_test.index[y_test != y_pred]  # original indices

for i in wrong_indices:
    row = df.iloc[i]
    flux = row.values[1:]  # exclude label column

    plt.figure(figsize=(10,4))
    plt.plot(flux)
    plt.title(f"Misclassified Example – True={y_test.loc[i]}, Pred={y_pred[list(X_test.index).index(i)]}")
    plt.xlabel("Time Index")
    plt


# In[22]:


get_ipython().system('pip install xgboost')


# In[23]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Train-test split (use smoothed data)
X = df_smooth2.drop(columns=[0])
y = df_smooth2[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost Model
model_xgb = XGBClassifier(
    eval_metric='logloss',          # required to avoid warning
    use_label_encoder=False,
    n_estimators=200,               # number of trees
    max_depth=5,                    # controls complexity
    learning_rate=0.05,             # smaller = safer learning
    subsample=0.8,                  # prevents overfitting
    colsample_bytree=0.8
)

model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))


# In[24]:


# Convert labels: 1 -> 0 (no planet), 2 -> 1 (planet)
df_smooth2[0] = df_smooth2[0].replace({1: 0, 2: 1})

df_smooth2[0].value_counts()


# In[25]:


from xgboost import XGBClassifier

# Train-test split
X = df_smooth2.drop(columns=[0])
y = df_smooth2[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_xgb = XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))


# In[27]:


df_test = pd.read_csv("exoplanet_ml_project/exotest.csv")


# In[28]:


df_test_smooth = df_test.apply(lambda row: smooth_flux(row), axis=1, result_type='expand')


# In[29]:


df_test_smooth[0] = df_test_smooth[0].replace({1: 0, 2: 1})


# In[30]:


X_real = df_test_smooth.drop(columns=[0])
y_real = df_test_smooth[0]

y_real_pred = model.predict(X_real)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_real, y_real_pred))
print(confusion_matrix(y_real, y_real_pred))


# In[31]:


from imblearn.over_sampling import SMOTE

X = df.drop("LABEL", axis=1)
y = df["LABEL"].replace({1: 0, 2: 1})  # convert labels

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

pd.Series(y_resampled).value_counts()


# In[ ]:




