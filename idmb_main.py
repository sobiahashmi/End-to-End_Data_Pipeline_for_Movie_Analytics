import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from joblib import dump


# Generate a synthetic dataset for movie analytics
n = 10000
df = pd.DataFrame({
    'MovieID': np.arange(n),
    'Title': [f'Movie {i}' for i in range(n)],
    'Genre': np.random.choice(['Drama', 'Comedy', 'Action', 'Horror'], size=n),
    'ReleaseYear': np.random.randint(1980, 2024, size=n),
    'Rating': np.round(np.random.normal(loc=6.5, scale=1.5, size=n), 1),
    'Votes': np.random.randint(100, 100000, size=n),
    'RevenueMillions': np.round(np.random.uniform(1, 300, size=n), 2)
})

print("Number of Records: ", df.shape[0])
print("Number of Columns: ", df.shape[1])

# convert to Binary Labels
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])

# Select Features
selected_features = ['MovieID', 'Genre', 'ReleaseYear', 'Rating', 'Votes']
X = df[selected_features]
y = df['RevenueMillions']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X,y)

# Save the model
dump(model,'imdb_pipeline_for_movie_analytics.joblib')