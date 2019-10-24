#!/usr/bin/env python
# coding: utf-8

# In[51]:


# Data from statsbomb and using pandas dataframes
import statsbomb as sb
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scikitplot as skplt
 
# Machine learning
from sklearn import preprocessing, model_selection, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from collections import Counter


# In[52]:


# Get all competitions
competitions = sb.Competitions()
 
# Get json data
json_data = competitions.data
 
# Convert to dataframe
df = competitions.get_dataframe()


# In[53]:


# Showing the competitions
# We will find and pick the data for World Cup 2018
df


# In[54]:


# Get all World Cup matches
wc_matches = sb.Matches(event_id = '43', season_id = '3').get_dataframe()


# In[55]:


# Show the World Cup games
wc_matches.head()


# In[56]:


# But we want to focus on shots and goals
# Create a list of all match ids for the matches in the 2018 World Cup 
match_list = wc_matches['match_id'].tolist()
 
# Create an empty dataframe to add all shots
shots_df = pd.DataFrame()
 
# Loop through and add all shots from every match to the empty dataframe
for i in match_list:
    events = sb.Events(event_id=str(i))
    shot = events.get_dataframe(event_type='shot')
    shots_df = shots_df.append(shot)


# In[57]:


# Details on our data
print(len(shots_df), "is the total number of shots from every match at the 2018 World Cup!")
print("--------------------")
 
print("Columns:")
print(list(shots_df))
print("--------------------")
 
print("Unique values:")
print(shots_df['type'].unique())
print("--------------------")
 
print("Unique values in the 'play_pattern' column:")
print(shots_df['play_pattern'].unique())


# In[58]:


# Remove penalties from the data since they are misleading in our analysis of shots
nopen_shots = shots_df[shots_df['type'] != 'Penalty']


# In[59]:


# Create a goal column, where 1 = goal and 0 = no goal
nopen_shots['goal'] = np.where(nopen_shots['outcome'] == 'Goal', 1, 0)


# In[60]:


# Show data
nopen_shots.head()


# In[61]:


# Calculating average shot conversion rate
attempts = len(nopen_shots)
goals = sum(nopen_shots['goal'])
misses = attempts - goals
conversion_rate = goals / attempts
print('Average conversion rate: ',round(conversion_rate*100,2),"%")


# In[62]:


# Show graph
# We see that most shots end up not being being a goal
sns.set(style="ticks", color_codes=True)
sns.countplot(x="goal", data=nopen_shots)


# In[64]:


# Feature engineering - Make new variables out of the raw data to improve model
# Resetting index
nopen_shots = nopen_shots.reset_index().drop('level_0', axis=1)
 
# Our new variables: distance
# Length of soccer field is 120 units long and width is 80 units wide
# Use distance formula and angle from the centre of the goal being scored on ,at (120, 40), to location of shot
nopen_shots['x_distance'] = 120 - nopen_shots['start_location_x']
nopen_shots['y_distance'] = abs(40 - nopen_shots['start_location_y'])
nopen_shots['distance'] = np.sqrt((nopen_shots['x_distance']**2 + nopen_shots['y_distance']**2))

# If we know the player's weak and strong foot, we can make look at these data individually
nopen_shots['body_part'] = np.where((nopen_shots['body_part'] == 'Right Foot')
                                 | (nopen_shots['body_part'] == 'Left Foot'), 'foot',
                                np.where(nopen_shots['body_part'] == 'Head', 'head', 'other'))


# In[66]:


# Find model features in cols
feature_cols = ['play_pattern', 'under_pressure', 'body_part', 'technique', 'first_time',
                'follows_dribble', 'redirect', 'one_on_one', 'open_goal', 'deflected', 'distance']


features = nopen_shots[feature_cols]
# Find model labels
labels = nopen_shots['goal']
 
# 0 to NA values
features = features.fillna(0)
labels = labels.fillna(0)


# In[67]:


# Show features
# As you can see, we have the new variables of distance and angle and have filled up NA values with 0
# We have the relevant and useful information we need
features


# In[68]:


# Label Encoding
# Make all categorical features to discrete values
cat_cols = ['play_pattern', 'under_pressure', 'body_part', 'technique', 'first_time',
                'follows_dribble', 'redirect', 'one_on_one', 'open_goal', 'deflected']
 
cat_features = features[cat_cols]
features = features.drop(cat_cols, axis=1)
features.head()


# In[69]:


# Discrete values for each category
le = preprocessing.LabelEncoder()
cat_features = cat_features.apply(le.fit_transform)

# Merge
features = features.merge(cat_features, left_index=True, right_index=True)
features.head()
# Now we have turned all categorical features into discrete values, so we can work with a consistent dataset


# In[70]:


# X = model features, and y = labels
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, shuffle=True, random_state=42)

# Using the Decision Tree Model
clf = DecisionTreeClassifier(random_state=42)
 
# Training
clf.fit(X_train, y_train)
 
# Making predictions
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)


# In[81]:


# Print results
print("Predicted goals from test data:", sum(y_pred))
print("xG:", "{0:.2f}".format(sum(y_pred_prob[:,1])))
print("Actual goals from test set):", sum(y_test))
print('Difference from my xG value and actual goals = ','{0:.2f}'.format(sum(y_pred_prob[:,1])-sum(y_test)))
print('')
print(metrics.classification_report(y_test, y_pred))
# Precision means of all the goals the modelled claimed to be a goal, how many were goals?
# Recall means of all the actual goals, how many did the model predict to be a goal?
# F1 score is the average of precision and recall, so the higher the F1 score, the better the model
 
# Find the most important feature for goal
important_features = pd.DataFrame({'feature':features.columns,'importance':np.round(clf.feature_importances_,3)})
important_features = important_features.sort_values('importance',ascending=False)
 
f, ax = plt.subplots(figsize=(12, 14))
g = sns.barplot(x='importance', y='feature', data=important_features,
                color="red", saturation=.2, label="Total")
g.set(xlabel='Feature Importance', ylabel='Feature', title='Importance of Features in xG Model')
plt.show()


# In[ ]:




