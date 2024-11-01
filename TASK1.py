# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('C:/Users/Sachin/Downloads/Clean_Dataset.csv/Flight_Dataset.csv', encoding='latin-1')

print(df.head())

print(df.columns)

print(df.info())

print(df.describe())


# Drop redundant columns
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])


# Display dataset information and check for missing values
print("Dataset Information:\n")
df.info()
print("\nMissing Values in Each Column:\n", df.isnull().sum())
print("\nSample Data:\n", df.head())

#heatmap for null values in cloumns of a dataset
if df.isnull().sum().sum() > 0:
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()
else:
    print("No missing values in the dataset.")


# Step 2: Feature Engineering
# Ensure duration is numeric and to calculate price per hour, you can divide the total price by the total duration
df['duration'] = df['duration'].astype(float)
df['price_per_hour'] = df['price'] / df['duration']


index1=df.destination_city.value_counts().index
###Index(['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'], dtype='object', name='destination_city')
values1=df.destination_city.value_counts().values
###[59097 57360 51068 49534 42726 40368]

plt.pie(values1,labels=index1)
plt.show()

# Step 3: Univariate Analysis - Numerical Features
# Plot distributions with improved labeling and clarity
fig, axes = plt.subplots(2,2, figsize=(10, 6))

# Price Distribution
#A histogram is used here because it’s ideal for showing the distribution of continuous data.
sns.histplot(df['price'], bins=30, kde=True, color='skyblue', ax=axes[0,0])
axes[0,0].set_title('Flight Price Distribution')
axes[0,0].set_xlabel('Price (INR)')
axes[0,0].set_ylabel('Number Of Flights')

# Duration Distribution
sns.histplot(df['duration'], bins=30, kde=True, color='lightgreen', ax=axes[0,1])
axes[0,1].set_title('Flight Duration Distribution')
axes[0,1].set_xlabel('Duration (hours)')
axes[0,1].set_ylabel('Number Of Flights')

# Days Left Distribution
sns.histplot(df['days_left'], bins=30, kde=True, color='coral', ax=axes[1,0])
axes[1,0].set_title('Booking Days Left Distribution')
axes[1,0].set_xlabel('Days Remaining to Departure')
axes[1,0].set_ylabel('Number Of Flights')

# Price per Hour Distribution
sns.histplot(df['price_per_hour'], bins=30, kde=True, color='purple',ax=axes[1,1])
axes[1,1].set_title('Price per Hour Distribution')
axes[1,1].set_xlabel('Price per Hour (INR)')
axes[1,1].set_ylabel('Frequency')

#Adjust the padding between and around subplots.
plt.tight_layout()
plt.show()


# Step 4: Univariate Analysis - Categorical Features
# Plot airline frequency with labeled count
plt.figure(figsize=(10,4))
sns.countplot(data=df, x='airline', order=df['airline'].value_counts().index, palette='Set3',width=0.4)
plt.title('Number of Flights by Airline')
plt.xlabel('Airline')
plt.ylabel('Number of Flights')
for index, value in enumerate(df['airline'].value_counts()):
    plt.text(value, index, str(value), ha='center')
plt.show()

# Plot class frequency with labels
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='class', palette='Set2',width=0.3)
plt.title('Flights by Class')
plt.xlabel('Flight Classes')
plt.ylabel('Number Of Flight')
for index, value in enumerate(df['class'].value_counts()):
    plt.text(index, value, str(value), ha='center')
plt.show()

# Step 5: Bivariate Analysis - Relationships Between Features
# Price vs Duration with Class and enhanced labeling
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='duration', y='price', hue='class', palette='cool', alpha=0.7)
plt.title('Price vs Duration by Flight Class')
plt.xlabel('Duration (hours)')
plt.ylabel('Price (INR)')
plt.legend(title='Class', loc='upper left')
plt.show()

# Boxplot for Price by Airline with labeled quartiles
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='airline', x='price', palette='Set3')
plt.title('Price Distribution by Airline')
plt.xlabel('Price (INR)')
plt.ylabel('Airline')

plt.show()

# Price vs Days Left with annotations for decreasing trend
plt.figure(figsize=(8, 4))
sns.scatterplot(data=df, x='days_left', y='price', color='green', alpha=0.5)
plt.title('Price vs Days Left Until Departure')
plt.xlabel('Days Left')
plt.ylabel('Price (INR)')
plt.show()

# Step 6: Multivariate Analysis - Airline, Class, and Stops Combined Effect on Price
# Facet grid for Price distribution across Airlines, Class, and Stops
g = sns.FacetGrid(df, col="class", row="stops", height=4, aspect=1.5)
g.map(sns.boxplot, "airline", "price", order=df['airline'].value_counts().index, palette="Set2")
g.set_xticklabels(rotation=0)
g.set_titles(col_template="Class: {col_name}", row_template="Stops: {row_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Price Distribution by Airline, Class, and Stops')
plt.show()

# Step 7: Correlation Analysis
# Correlation heatmap with enhanced readability
plt.figure(figsize=(8, 4))
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.xticks(rotation=0, ha='right')
plt.yticks(rotation=0)
plt.show()

# Step 8: Summary Statistics and Additional Analysis
# Displaying summary statistics for numeric features
print("\nSummary Statistics for Numerical Columns:\n", df.describe())

# Mean Price per Hour by Airline and Class with clearer indexing
price_per_hour_summary = df.groupby(['airline', 'class'])['price_per_hour'].mean().sort_values(ascending=False)
print("\nMean Price per Hour by Airline and Class:\n", price_per_hour_summary)

# Visualization of average price per hour by airline
plt.figure(figsize=(8, 4))
price_per_hour_summary.unstack().plot(kind='bar', stacked=True, color=['#FFA07A', '#20B2AA'])
plt.title('Average Price per Hour by Airline and Flight Class')
plt.xlabel('Airline')
plt.ylabel('Average Price per Hour (INR)')
plt.xticks(ha='right')
plt.legend(title="Class", loc="upper right")
plt.show()
