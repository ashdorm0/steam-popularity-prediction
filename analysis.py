# Name: Thomas Radulescu | Student number: C00302068
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

DATA = Path('data/steam.csv')
# Load only the columns needed for the popularity analysis.
df = pd.read_csv(DATA)
print('Original shape:', df.shape)
print(df[['appid', 'name', 'steamspy_tags', 'owners']].head().to_string(index=False))

df = df[['appid', 'name', 'steamspy_tags', 'owners']].copy()
# Remove incomplete, duplicate, or unusable game records before modelling.
print('Missing values before cleaning:')
print(df.isna().sum().to_string())
print('Duplicate app IDs:', df['appid'].duplicated().sum())
before = len(df)
df = df.dropna(subset=['appid', 'steamspy_tags', 'owners']).drop_duplicates('appid')
df = df[df['steamspy_tags'].str.strip().ne('')].copy()
# Keep owner ranges that match the expected "lower-upper" format and order.
bounds = df['owners'].str.extract(r'^(\d+)-(\d+)$').apply(pd.to_numeric)
valid = bounds.notna().all(axis=1) & (bounds[0] <= bounds[1])
df = df.loc[valid].copy()
df['owners_lower'] = bounds.loc[valid, 0].astype(int)


# Convert the lower end of each owner range into the target class.
def popularity(lower):
    if lower < 20000:
        return 'Low'
    if lower < 100000:
        return 'Medium'
    return 'High'
df['popularity'] = df['owners_lower'].apply(popularity)
print('Rows removed:', before - len(df))
print('Rows retained:', len(df))
labels = ['Low', 'Medium', 'High']
print(df['popularity'].value_counts().reindex(labels).to_string())

# Stratification keeps the popularity mix similar in both data splits.
train, test = train_test_split(df, test_size=0.2, random_state=42,
                              stratify=df['popularity'])
print('Training rows:', len(train), 'Test rows:', len(test))

# Inspect the class distribution used to train the model.
train['popularity'].value_counts().reindex(labels).plot.bar(rot=0)
plt.ylabel('Number of games')
plt.title('Popularity categories in the training data')
plt.tight_layout()
plt.show()

# Explore which tags occur most often among the high-popularity games.
high_tags = train.loc[train['popularity'].eq('High'), 'steamspy_tags'].str.split(';').explode()
print('Most common tags among high-ownership training games:')
print(high_tags.value_counts().head(10).to_string())
high_tags.value_counts().head(10).sort_values().plot.barh()
plt.xlabel('Number of high-ownership games')
plt.title('Common tags among high-ownership training games')
plt.tight_layout()
plt.show()

# Encode each semicolon-separated tag as a binary feature.
X_train = train['steamspy_tags'].str.get_dummies(sep=';')
# Match test columns to training columns so the model sees the same features.
X_test = test['steamspy_tags'].str.get_dummies(sep=';').reindex(columns=X_train.columns, fill_value=0)
y_train = train['popularity']
y_test = test['popularity']

# Limit tree complexity to reduce overfitting on sparse tag features.
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('Tag features:', X_train.shape[1])
print('Training accuracy:', round(model.score(X_train, y_train), 4))
print('Test accuracy:', round(accuracy_score(y_test, predictions), 4))
# Compare the model with the simple strategy of always choosing the
# most frequent training class.
majority = y_train.mode()[0]
baseline = (y_test == majority).mean()
print('Always predict', majority, 'baseline accuracy:', round(baseline, 4))
print(classification_report(y_test, predictions, labels=labels, zero_division=0))
# Show which actual and predicted popularity classes are most often confused.
ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=labels, cmap='Blues')
plt.title('Decision tree test results')
plt.tight_layout()
plt.show()