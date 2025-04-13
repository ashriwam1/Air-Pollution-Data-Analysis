import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("AirPollution.csv")

print(df.head())
print(df.tail())

print(df.info())
print(df.describe())

print(df.isnull().sum())

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print(df.isnull().sum())


df.to_csv("cleanedAir.csv", index=False)

print("Columns:", df.columns)
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
    
plt.figure(figsize=(10, 8))

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Numeric Features")
plt.show()

pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
for col in pollutants:
    if col in df.columns:
        plt.figure()
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
if 'City' in df.columns:
    city_pollution = df.groupby('City')[pollutants].mean().sort_values(by='PM2.5', ascending=False)
    print(city_pollution.head())

    city_pollution['PM2.5'].plot(kind='bar', figsize=(12,6), color='skyblue')
    plt.title("Average PM2.5 Levels by City")
    plt.ylabel("PM2.5")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def classify_pm25(value):
    if value <= 50:
        return 'Good'
    elif value <= 100:
        return 'Moderate'
    elif value <= 200:
        return 'Poor'
    else:
        return 'Very Poor'

if 'PM2.5' in df.columns:
    df['PM2.5_Level'] = df['PM2.5'].apply(classify_pm25)
    print(df[['PM2.5', 'PM2.5_Level']].head())
eda_summary = df.describe().transpose()
eda_summary.to_csv("EDA_Summary.csv")



plt.style.use('ggplot')


if 'City' in df.columns and 'PM2.5' in df.columns:
    city_avg = df.groupby('City')['PM2.5'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    city_avg.plot(kind='bar', color='skyblue')
    plt.title("Average PM2.5 per City")
    plt.ylabel("PM2.5 Level")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
for col in pollutants:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        plt.hist(df[col], bins=30, color='lightcoral', edgecolor='black')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

#  Box Plot to Detect Outliers
for col in pollutants:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Heatmap of Pollutants")
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')
plt.tight_layout()
plt.show()

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    plt.figure(figsize=(12, 6))
    df.resample('M')['PM2.5'].mean().plot(label='PM2.5')
    if 'PM10' in df.columns:
        df.resample('M')['PM10'].mean().plot(label='PM10')
    plt.title("Monthly Average PM2.5 and PM10")
    plt.xlabel("Date")
    plt.ylabel("Pollution Level")
    plt.legend()
    plt.tight_layout()
    plt.show()

def classify_pm25(value):
    if value <= 50:
        return 'Good'
    elif value <= 100:
        return 'Moderate'
    elif value <= 200:
        return 'Poor'
    else:
        return 'Very Poor'

if 'PM2.5' in df.columns:
    df['PM2.5_Level'] = df['PM2.5'].apply(classify_pm25)
    pie_data = df['PM2.5_Level'].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=['green', 'yellow', 'orange', 'red'])
    plt.title("PM2.5 Level Distribution")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

sns.set_style("whitegrid")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns


if 'City' in df.columns and 'PM2.5' in df.columns:
    city_avg = df.groupby('City')['PM2.5'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(city_avg.index, city_avg.values, color='skyblue')
    plt.title("Average PM2.5 by City")
    plt.ylabel("PM2.5")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    plt.hist(df[col].dropna(), bins=30, color='lightcoral', edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color='lightblue')
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()


scatter_pairs = [('PM2.5', 'PM10'), ('NO2', 'SO2'), ('CO', 'O3')]
for x, y in scatter_pairs:
    if x in df.columns and y in df.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[x], y=df[y], alpha=0.6)
        plt.title(f"Scatter Plot: {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.show()


if len(numeric_cols) > 1:
    sns.pairplot(df[numeric_cols].dropna(), diag_kind="kde", corner=True)
    plt.suptitle("Pairplot of Numeric Features", y=1.02)
    plt.show()

#HEATMAP
plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#  PIE CHART for PM2.5 level classification
def classify_pm25(val):
    if val <= 50:
        return "Good"
    elif val <= 100:
        return "Moderate"
    elif val <= 200:
        return "Poor"
    else:
        return "Very Poor"

if 'PM2.5' in df.columns:
    df['PM2.5_Level'] = df['PM2.5'].apply(classify_pm25)
    pie_counts = df['PM2.5_Level'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(pie_counts.values, labels=pie_counts.index, autopct='%1.1f%%',
            colors=['green', 'yellow', 'orange', 'red'], startangle=140)
    plt.title("PM2.5 Level Distribution")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

#  LINE PLOT over time
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')
    monthly_avg = df.resample('M').mean(numeric_only=True)
    monthly_avg[['PM2.5', 'PM10']].plot(figsize=(12, 6), marker='o')
    plt.title("Monthly Average PM2.5 and PM10")
    plt.ylabel("Pollution Level")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Detecting and handling outliers
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)]
        print(f"{col}: {len(outliers)} outliers detected")

        
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]

    return cleaned_df

df_no_outliers = remove_outliers_iqr(df, numeric_cols)

df_no_outliers.to_csv("cleanedAir_no_outliers.csv", index=False)
print("Outliers removed. Clean dataset saved as 'cleanedAir_no_outliers.csv'")

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

if 'City' in df.columns and 'PM2.5' in df.columns:
    city_avg = df.groupby('City')['PM2.5'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(city_avg.index, city_avg.values, color='skyblue')
    plt.title("Average PM2.5 by City")
    plt.ylabel("PM2.5")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

pairs = [('PM2.5', 'PM10'), ('NO2', 'SO2'), ('CO', 'O3')]
for x, y in pairs:
    if x in df.columns and y in df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[x], df[y], alpha=0.6, c='green')
        plt.title(f"{x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.show()

def classify_pm25(val):
    if val <= 50:
        return 'Good'
    elif val <= 100:
        return 'Moderate'
    elif val <= 200:
        return 'Poor'
    else:
        return 'Very Poor'

if 'PM2.5' in df.columns:
    df['PM2.5_Level'] = df['PM2.5'].apply(classify_pm25)
    level_counts = df['PM2.5_Level'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
            colors=['green', 'yellow', 'orange', 'red'], startangle=140)
    plt.title("PM2.5 Level Distribution")
    plt.tight_layout()
    plt.axis("equal")
    plt.show()

# 6. LINE PLOT - Trend over time
if 'Date' in df.columns:
    df_time = df.set_index('Date')
    monthly_avg = df_time.resample('M').mean(numeric_only=True)
    if 'PM2.5' in monthly_avg.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(monthly_avg.index, monthly_avg['PM2.5'], marker='o', color='purple')
        plt.title("Monthly Average PM2.5")
        plt.xlabel("Date")
        plt.ylabel("PM2.5")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


'''if len(pollutants) >= 2:
    monthly_avg[pollutants].fillna(0).plot.area(figsize=(12, 6), alpha=0.5)
    plt.title("Stacked Area Plot of Pollutants Over Time")
    plt.xlabel("Date")
    plt.ylabel("Levels")
    plt.tight_layout()
    plt.show()'''

import math
cols = 2
rows = math.ceil(len(numeric_cols) / cols)
plt.figure(figsize=(12, rows * 4))
for i, col in enumerate(numeric_cols):
    plt.subplot(rows, cols, i + 1)
    plt.hist(df[col], bins=20, color='steelblue', edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.tight_layout()
plt.show()

if 'City' in df.columns:
    stacked = df.groupby('City')[pollutants].mean().head(5)
    stacked.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set3')
    plt.title("Stacked Bar Chart of Pollutants (Top 5 Cities)")
    plt.ylabel("Pollutant Level")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

import numpy as np
if len(pollutants) >= 3:
    angles = np.linspace(0, 2 * np.pi, len(pollutants), endpoint=False).tolist()
    angles += angles[:1]
    values = df[pollutants].mean().tolist()
    values += values[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pollutants)
    ax.set_title("Average Pollutant Levels (Polar Chart)")
    plt.tight_layout()
    plt.show()

