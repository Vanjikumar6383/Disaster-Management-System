import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("flood_dataset_uncleaned.csv")

print("Shape:", df.shape)

# =========================
# BASIC CLEANING
# =========================

# remove spaces
obj_cols = df.select_dtypes(
    include=["object", "string"]
).columns

for col in obj_cols:
    df[col] = df[col].astype(str).str.strip()

# lowercase standardization
for col in ["river_name","state","embankment_condition","alert_level"]:
    if col in df.columns:
        df[col] = df[col].str.lower()

# Fix inconsistent embankment condition values
df["embankment_condition"] = df["embankment_condition"].replace({
    "god": "good",
    "gud": "good",
    "modrate": "moderate",
    "por": "poor"
})

# date conversion
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# numeric conversion
numeric_convert = [
    "discharge_cumecs",
    "distance_to_river_km"
]

for col in numeric_convert:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# flood occurred cleanup
df["flood_occurred"] = (
    df["flood_occurred"]
    .replace({
        "1":"yes",
        "0":"no",
        "Yes":"yes",
        "No":"no"
    })
    .str.lower()
)

# =========================
# MISSING VALUES
# =========================
print("\nMissing Values\n")
print(df.isnull().sum())

# Fix impossible negative population density
df.loc[
    df["population_density"] < 0,
    "population_density"
] = np.nan

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# =========================
# FLOOD RISK VALIDATION
# =========================

invalid_risk = df[
    (df["flood_risk_score"] < 0) |
    (df["flood_risk_score"] > 1)
]

print(
    "\nFlood Risk Scores Outside 0-1:",
    len(invalid_risk)
)
# =========================
# DUPLICATES
# =========================

print("\nDuplicates:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# =========================
# OUTLIER DETECTION
# =========================

outlier_report = {}

for col in num_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = len(
        df[
            (df[col] < lower) |
            (df[col] > upper)
        ]
    )

    outlier_report[col] = count

print("\nOutlier Report")
print(outlier_report)


# =========================
# OUTLIER HANDLING
# =========================

print("\nOutlier Handling")

# Create a flag column
df["outlier_flag"] = False

# Store outlier count
outlier_handling_report = {}

for col in num_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Identify outliers
    mask = (
        (df[col] < lower) |
        (df[col] > upper)
    )

    # Mark the record as an outlier
    df.loc[mask, "outlier_flag"] = True

    outlier_handling_report[col] = mask.sum()

print("\nOutlier Report:")
print(outlier_handling_report)

print(
    "\nTotal Suspicious/Outlier Records:",
    df["outlier_flag"].sum()
)

print(
    "\nOutliers are flagged for investigation."
)

print(
    "Extreme values are NOT automatically removed or capped."
)
# =========================
# RISK ANALYSIS
# =========================

print("\nFlood Risk Analysis")

risk_summary = df.groupby("alert_level")[
    [
        "rainfall_mm",
        "water_level_m",
        "flood_risk_score"
    ]
].mean()

print(risk_summary)

# =========================
# RELATIONSHIP ANALYSIS
# =========================

corr = df[num_cols].corr()

plt.figure(figsize=(16,10))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =========================
# INTERLINK ANALYSIS
# =========================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="rainfall_mm",
    y="water_level_m",
    hue="alert_level"
)

plt.title("Rainfall vs Water Level")
plt.show()

# =========================
# PATTERN IDENTIFICATION
# =========================
plt.figure(figsize=(8,6))

sns.barplot(
    data=df,
    x="alert_level",
    y="flood_risk_score",
    estimator="mean"
)

plt.title("Alert Level vs Average Flood Risk Score")
plt.xlabel("Alert Level")
plt.ylabel("Average Flood Risk Score")
plt.show()
 # =========================
# DESCRIPTIVE STATISTICS
# =========================

print("\nDescriptive Statistics")
print(df.describe())

print("\nCategorical Summary")
print(df.describe(include=["object", "string"]))

# =========================
# VISUALIZATION 1
# =========================

plt.figure(figsize=(8,5))

sns.histplot(
    df["rainfall_mm"],
    bins=30,
    kde=True
)

plt.title("Rainfall Distribution")
plt.show()

# =========================
# VISUALIZATION 2
# =========================

plt.figure(figsize=(8,5))

sns.histplot(
    df["water_level_m"],
    bins=30,
    kde=True
)

plt.title("Water Level Distribution")
plt.show()


# =========================
# UNIQUE FEATURE 1
# DISASTER DATA DETECTIVE
# =========================

print("\nDISASTER DATA DETECTIVE")

# Rule 1: Water level crossed danger level
# but alert level is still low
rule_1 = (
    (df["water_level_m"] > df["danger_level_m"]) &
    (df["alert_level"] == "low")
)

# Rule 2: Very high flood risk
# but flood did not occur
rule_2 = (
    (df["flood_risk_score"] > 0.8) &
    (df["flood_occurred"] == "no")
)

# Combine suspicious conditions
df["detective_status"] = "normal"

df.loc[rule_1, "detective_status"] = "needs_investigation"
df.loc[rule_2, "detective_status"] = "needs_investigation"

# Extract suspicious records
suspicious_records = df[
    df["detective_status"] == "needs_investigation"
]

print(
    "Suspicious Records Found:",
    suspicious_records.shape[0]
)

# Save report
suspicious_records.to_csv(
    "suspicious_records.csv",
    index=False
)

print("Disaster Data Detective Report Saved.")

# =========================
# UNIQUE FEATURE 2
# BEFORE DISASTER PATTERN
# =========================

print("\nBEFORE DISASTER PATTERN")

# Use only records with valid dates
temporal_df = df.dropna(subset=["date"]).copy()

# Sort by river and date
temporal_df = temporal_df.sort_values(
    ["river_name", "date"]
).reset_index(drop=True)

# Previous observation from the SAME river
temporal_df["previous_rainfall"] = (
    temporal_df
    .groupby("river_name")["rainfall_mm"]
    .shift(1)
)

temporal_df["previous_water_level"] = (
    temporal_df
    .groupby("river_name")["water_level_m"]
    .shift(1)
)

# Change from previous observation
temporal_df["rainfall_change"] = (
    temporal_df["rainfall_mm"] -
    temporal_df["previous_rainfall"]
)

temporal_df["water_level_change"] = (
    temporal_df["water_level_m"] -
    temporal_df["previous_water_level"]
)

# Identify high/severe events
severe_cases = temporal_df[
    temporal_df["alert_level"].isin(["high", "severe"])
]

# Conditions increasing before the event
before_disaster = severe_cases[
    (severe_cases["rainfall_change"] > 0) &
    (severe_cases["water_level_change"] > 0)
]

print(
    "High/Severe Events:",
    severe_cases.shape[0]
)

print(
    "Events with Rising Rainfall + Rising Water Level:",
    before_disaster.shape[0]
)

# Pattern summary
pattern_summary = before_disaster[
    [
        "rainfall_mm",
        "upstream_rainfall_mm",
        "forecast_rainfall_next24_mm",
        "water_level_m",
        "soil_moisture_pct",
        "flood_risk_score"
    ]
].mean()

print("\nBefore Disaster Pattern:")
print(pattern_summary)

# Visualization
plt.figure(figsize=(8,5))

pattern_summary.plot(kind="bar")

plt.title("Before Disaster Pattern")
plt.ylabel("Average Value")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# =========================
# UNIQUE FEATURE 3
# HIDDEN COMBINATION FINDER
# =========================

print("\nHIDDEN COMBINATION FINDER")

combo = df.groupby(
[
"alert_level",
"embankment_condition"
]
)[
[
"rainfall_mm",
"water_level_m",
"flood_risk_score"
]
].mean()

print(combo)

# =========================
# ADVANCED COMBINATION
# =========================

high_risk_combo = df[
(
df["rainfall_mm"] >
df["rainfall_mm"].quantile(0.75)
)
&
(
df["water_level_m"] >
df["water_level_m"].quantile(0.75)
)
&
(
df["soil_moisture_pct"] >
df["soil_moisture_pct"].quantile(0.75)
)
]

print(
"\nHigh Risk Combination Records:",
high_risk_combo.shape[0]
)

# =========================
# SAVE CLEAN DATA
# =========================

df.to_csv(
    "EDA_Cleaned_Dataset.csv",
    index=False
)

print("\nEDA COMPLETED SUCCESSFULLY")