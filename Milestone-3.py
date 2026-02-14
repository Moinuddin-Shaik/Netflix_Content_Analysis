# Databricks notebook source
# DBTITLE 1,Import Libraries
# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, year
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# COMMAND ----------

# DBTITLE 1,Load Dataset
from pyspark.sql import SparkSession
path = "/Volumes/workspace/default/netflix/feature_engineering_/"

df = spark.read.csv(path, header=True, inferSchema=True)


display(df)

# COMMAND ----------

# DBTITLE 1,Columns in dataset
# DBTITLE 1,Check column names
df.columns


# COMMAND ----------

from pyspark.sql.functions import col, expr, year

# Safely parse 'date_added' column into a valid date (invalid ones -> NULL)
df = df.withColumn(
    "date_added_parsed",
    expr("try_to_date(date_added, 'MMMM d, yyyy')")
)

# Extract year from parsed dates
df = df.withColumn(
    "release_year_extracted",
    year(col("date_added_parsed"))
)

# Use try_cast to safely convert 'release_year' to integer (malformed -> NULL)
df = df.withColumn("release_year", expr("try_cast(release_year AS int)"))

# Fill missing or invalid values with a default year (e.g., 2000)
df = df.fillna({"release_year": 2000, "release_year_extracted": 2000})

# Display results to confirm cleaning
display(df.select("date_added", "date_added_parsed", "release_year", "release_year_extracted"))


# COMMAND ----------

from pyspark.sql.functions import col

# Handle NULLs in numeric columns before VectorAssembler
numeric_cols = [
    "movie", "tv_show", "Independent_Movies", "Romantic_TV_Shows", "Thrillers",
    "Dramas", "Docuseries", "Sports_Movies", "Horror_Movies", "Cult_Movies",
    "Classic_Movies", "Anime_Features", "Stand_Up_Comedy_and", "Crime_TV_Shows",
    "TV_Sci_Fi_and_Fantasy", "Faith_and_Spiritua", "duration_min",
    "seasons_num", "country_count", "release_year"
]

# Fill NULLs with 0 for all numeric columns
df = df.fillna(0, subset=numeric_cols)

# Re-run VectorAssembler with handleInvalid='keep'
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol="features",
    handleInvalid="keep"  # this ignores or keeps invalid/nulls safely
)

df_features = assembler.transform(df)

display(df_features.select("features", "release_year", "duration_min").limit(5))


# COMMAND ----------

# DBTITLE 1,Clustering
from pyspark.ml.clustering import KMeans

# Initialize KMeans with 3 clusters
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=4, seed=42)

# Fit the model
kmeans_model = kmeans.fit(df_features)

# Add the predicted cluster column to the dataframe
df_clustered = kmeans_model.transform(df_features)

# Show few samples with cluster assignments
display(df_clustered.select("title", "release_year", "duration_min", "country_count", "cluster"))


# COMMAND ----------

# DBTITLE 1,Cluster Count
from pyspark.sql.functions import col

# Count number of items in each cluster
cluster_counts = df_clustered.groupBy("cluster").count()
cluster_counts.show()


# COMMAND ----------

# Convert Spark DataFrame to Pandas for plotting
cluster_counts_pd = cluster_counts.toPandas()

import matplotlib.pyplot as plt
import seaborn as sns

# Bar plot
plt.figure(figsize=(8,5))
sns.barplot(x='cluster', y='count', data=cluster_counts_pd, palette='Set2')
plt.title("Number of Shows/Movies in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()


# COMMAND ----------

# DBTITLE 1,Add cluster labels to your DF
# Assuming kmeans_model is your trained KMeans model
df_clusters = kmeans_model.transform(df_features)  # df_features contains your 'features' column
df_clusters.select("show_id", "title", "cluster").show(10000)


# COMMAND ----------

# DBTITLE 1,Cast boolean columns to integer
from pyspark.sql.functions import col

bool_cols = ["Independent_Movies", "Romantic_TV_Shows", "Thrillers", "TV_Mysteries", "TV_Horror"]

for c in bool_cols:
    df_clusters = df_clusters.withColumn(c, col(c).cast("integer"))


# COMMAND ----------

# DBTITLE 1,Compute average per cluster
from pyspark.sql.functions import avg

numeric_features = ["movie", "tv_show", "Independent_Movies", "Romantic_TV_Shows",
                    "Thrillers", "Dramas", "Docuseries", "Sports_Movies", "Horror_Movies",
                    "Cult_Movies", "Classic_Movies", "Anime_Features", "Stand_Up_Comedy_and",
                    "Crime_TV_Shows", "TV_Sci_Fi_and_Fantasy", "Faith_and_Spiritua",
                    "duration_min", "seasons_num", "country_count", "release_year"]

cluster_summary = df_clusters.groupBy("cluster").agg(
    *[avg(f).alias(f"avg_{f}") for f in numeric_features]
)

display(cluster_summary)


# COMMAND ----------

# DBTITLE 1,Visualize Cluster Averages
import matplotlib.pyplot as plt
import seaborn as sns

# Convert Spark DataFrame to Pandas for visualization
cluster_summary_pd = cluster_summary.toPandas()

# Set cluster as index
cluster_summary_pd.set_index("cluster", inplace=True)

# Select only important features for readability
features_to_plot = ["avg_duration_min", "avg_country_count", "avg_release_year"]

# Melt the dataframe to long format for Seaborn
cluster_summary_melted = cluster_summary_pd[features_to_plot].reset_index().melt(id_vars="cluster", var_name="Feature", value_name="Average")

# Set the plotting style
sns.set(style="whitegrid", palette="pastel")

plt.figure(figsize=(10, 6))
sns.barplot(
    data=cluster_summary_melted,
    x="cluster",
    y="Average",
    hue="Feature",
    palette="viridis",
    edgecolor="black"
)

plt.title("📊 Average Feature Values per Cluster", fontsize=15, fontweight='bold')
plt.xlabel("Cluster Label", fontsize=12)
plt.ylabel("Average Value", fontsize=12)
plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# COMMAND ----------

# DBTITLE 1,Genre Distribution Per Cluster
# Import required libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Convert to pandas
cluster_summary_pd = cluster_summary.toPandas()

# Set the cluster column as index
cluster_summary_pd.set_index("cluster", inplace=True)

# Select only genre-related averages for the heatmap
genre_cols = [
    "avg_Independent_Movies", "avg_Romantic_TV_Shows", "avg_Thrillers", 
    "avg_Dramas", "avg_Docuseries", "avg_Sports_Movies", "avg_Horror_Movies", 
    "avg_Cult_Movies", "avg_Classic_Movies", "avg_Anime_Features", 
    "avg_Stand_Up_Comedy_and", "avg_Crime_TV_Shows", 
    "avg_TV_Sci_Fi_and_Fantasy", "avg_Faith_and_Spiritua"
]

# Create a heatmap
plt.figure(figsize=(14, 7))
sns.heatmap(
    cluster_summary_pd[genre_cols],
    cmap="Greens",  # You can try "Blues", "YlGnBu", etc.
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Average Genre Value"}
)

# Beautify
plt.title("Genre Distribution per Cluster", fontsize=16, fontweight="bold", color="#222222")
plt.xlabel("Genres", fontsize=12)
plt.ylabel("Cluster", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# COMMAND ----------

# DBTITLE 1,Interpret Each Cluster
# Convert summary to pandas (if not already)
cluster_summary_pd = cluster_summary.toPandas()
cluster_summary_pd.set_index("cluster", inplace=True)

# Display the summary to interpret values
display(cluster_summary_pd.round(2))

# Example: Manual interpretation (you can print or write this as markdown)
for cluster_id, row in cluster_summary_pd.iterrows():
    print(f"\nCluster {cluster_id} Interpretation:")
    print("-" * 40)
    
    # Identify top 3 genres for this cluster
    top_genres = row.filter(like="avg_").sort_values(ascending=False).head(3)
    print(f"Top 3 Strong Genres: {', '.join(top_genres.index.str.replace('avg_', '').tolist())}")
    
    # Add more descriptive logic
    if row["avg_duration_min"] > 90:
        print("These titles tend to have longer durations, indicating full-length movies.")
    elif row["avg_seasons_num"] > 1:
        print("This cluster mainly contains multi-season TV shows.")
    else:
        print("This cluster includes short or single-season content.")
    
    # Release year pattern
    if row["avg_release_year"] > 2015:
        print("Most of the content in this cluster is recent (post-2015).")
    else:
        print("This cluster features older content.")


# COMMAND ----------

# DBTITLE 1,Summarize Each  Cluster
# Count of titles per cluster
cluster_counts = df_clusters.groupBy("cluster").count().orderBy("cluster")
display(cluster_counts)

# Average numeric feature values (for better understanding)
display(cluster_summary)

# Top genres per cluster (based on average)
for col_name in ["Independent_Movies", "Romantic_TV_Shows", "Thrillers", "Dramas", "Docuseries"]:
    print(f"\n🔹 Cluster-wise average for {col_name}:")
    df_clusters.groupBy("cluster").agg(avg(col_name).alias(f"avg_{col_name}")).orderBy("cluster").show()


# COMMAND ----------

# DBTITLE 1,Random Forest Classification
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Define numeric input features
feature_cols = ["movie", "tv_show", "Independent_Movies", "Romantic_TV_Shows",
                "Thrillers", "Dramas", "Docuseries", "Sports_Movies", "Horror_Movies",
                "Cult_Movies", "Classic_Movies", "Anime_Features", "Stand_Up_Comedy_and",
                "Crime_TV_Shows", "TV_Sci_Fi_and_Fantasy", "Faith_and_Spiritua",
                "duration_min", "seasons_num", "country_count", "release_year"]

# Assemble features into a new column
assembler = VectorAssembler(inputCols=feature_cols, outputCol="rf_features")

# Define Random Forest Classifier
rf = RandomForestClassifier(labelCol="cluster", featuresCol="rf_features", numTrees=20, seed=42)

# Build pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Split the data
train_df, test_df = df_clusters.randomSplit([0.8, 0.2], seed=42)

# Fit the model
rf_model = pipeline.fit(train_df)

# Make predictions
predictions = rf_model.transform(test_df)

# Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="cluster", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"✅ Random Forest Classification Accuracy: {accuracy:.4f}")


# COMMAND ----------

# DBTITLE 1,Forest feature Prediction
# Extract feature importances from the Random Forest model
rf_model_stage = rf_model.stages[-1]  # RandomForestClassifier is the last stage
importances = rf_model_stage.featureImportances

# Create a pandas DataFrame for visualization
import pandas as pd
import matplotlib.pyplot as plt

feature_importance_pd = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances.toArray()
}).sort_values(by="importance", ascending=False)

# Plot
plt.figure(figsize=(12,6))
plt.barh(feature_importance_pd["feature"], feature_importance_pd["importance"], color="darkgreen")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importances for Cluster Prediction")
plt.gca().invert_yaxis()  # Highest importance on top
plt.show()


# COMMAND ----------

# DBTITLE 1,Prepare the DF for classification
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

# Select numeric features for clustering
numeric_features = ["movie", "tv_show", "Independent_Movies", "Romantic_TV_Shows",
                    "Thrillers", "Dramas", "Docuseries", "Sports_Movies", "Horror_Movies",
                    "Cult_Movies", "Classic_Movies", "Anime_Features", "Stand_Up_Comedy_and",
                    "Crime_TV_Shows", "TV_Sci_Fi_and_Fantasy", "Faith_and_Spiritua",
                    "duration_min", "seasons_num", "country_count", "release_year"]

# Convert booleans to integers
for col_name in ["Independent_Movies", "Romantic_TV_Shows", "Thrillers"]:
    df = df.withColumn(col_name, df[col_name].cast("int"))

# Assemble features
assembler = VectorAssembler(inputCols=numeric_features, outputCol="features")
df_features = assembler.transform(df)

# Train KMeans
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=3, seed=42)
kmeans_model = kmeans.fit(df_features)
df_clusters = kmeans_model.transform(df_features)  # <-- now this has the "cluster" column


# COMMAND ----------

# DBTITLE 1,Train Random Forest Classifier
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",  # or create a new VectorAssembler if you want
    labelCol="cluster",
    predictionCol="prediction"
)

rf_model = rf.fit(df_clusters)
df_rf_predictions = rf_model.transform(df_clusters)


# COMMAND ----------

# MAGIC %md
# MAGIC ACCURACY = (TP+TN) / (TP+TN+FP+FN)
# MAGIC PRECISION = TP / (TP + FP)
# MAGIC RECALL = TP / (TP + FN)
# MAGIC F1-SCORE = 2 * (PR*RE) / (PR+RE)
# MAGIC

# COMMAND ----------

# DBTITLE 1,Evaluate Metrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

accuracy = MulticlassClassificationEvaluator(
    labelCol="cluster", predictionCol="prediction", metricName="accuracy"
).evaluate(df_rf_predictions)

precision = MulticlassClassificationEvaluator(
    labelCol="cluster", predictionCol="prediction", metricName="weightedPrecision"
).evaluate(df_rf_predictions)

recall = MulticlassClassificationEvaluator(
    labelCol="cluster", predictionCol="prediction", metricName="weightedRecall"
).evaluate(df_rf_predictions)

f1 = MulticlassClassificationEvaluator(
    labelCol="cluster", predictionCol="prediction", metricName="f1"
).evaluate(df_rf_predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# COMMAND ----------

# Split the data into training and testing sets
train_data, test_data = df_clusters.randomSplit([0.8, 0.2], seed=42)

print("Training Data Count:", train_data.count())
print("Test Data Count:", test_data.count())


# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.classification import RandomForestClassifier

# Create a label column for classification (1 = Movie, 0 = TV Show)
df_rf = df_clusters.withColumn("label", col("movie").cast("double"))

# Split into training and test sets
train_data, test_data = df_rf.randomSplit([0.8, 0.2], seed=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50)
rf_model = rf.fit(train_data)

# Make predictions on test data
rf_predictions = rf_model.transform(test_data)


# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Convert predictions to pandas for visualization
preds_df = rf_predictions.select("label", "prediction").toPandas()

# Confusion Matrix
cm = confusion_matrix(preds_df["label"], preds_df["prediction"])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="BuGn")
plt.title("Confusion Matrix - Random Forest Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification Report (Accuracy, Precision, Recall, F1)
print("\nClassification Report:\n")
print(classification_report(preds_df["label"], preds_df["prediction"]))


# COMMAND ----------

countries = [
    "Thailand", "Poland", "Italy", "Belgium", "Germany", "Denmark", "Sweden", "Japan",
    "Nigeria", "Philippines", "Netherlands", "Norway", "Finland", "Canada", "South_Africa",
    "Spain", "Mexico", "Russia", "France", "Australia", "Indonesia", "Turkey", "USA",
    "Egypt", "China", "India", "Brazil", "Argentina", "Switzerland", "Pakistan",
    "India", "india", "INDIA"  # example duplicates
]

# Remove duplicates (case-insensitive)
unique_countries = sorted(set([c.strip().capitalize() for c in countries]))
count=len(countries)

print(unique_countries)
print("Number of countries:", count)


# COMMAND ----------

# DBTITLE 1,Define your country and genre columns
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ✅ Country columns (cleaned and correct)
countries = [
    'Argentina', 'Australia', 'Belgium', 'Brazil', 'Canada', 'China', 'Denmark',
    'Egypt', 'Finland', 'France', 'Germany', 'India', 'Indonesia', 'Italy', 'Japan',
    'Mexico', 'Netherlands', 'Nigeria', 'Norway', 'Pakistan', 'Philippines', 'Poland',
    'Russia', 'South_Africa', 'Spain', 'Sweden', 'Switzerland', 'Thailand', 'Turkey', 'USA'
]

# ✅ Genre columns (based on your schema)
genres = [
    'Independent_Movies', 'Romantic_TV_Shows', 'Thrillers', 'Dramas', 'Docuseries',
    'Sports_Movies', 'Horror_Movies', 'Cult_Movies', 'TV_Mysteries', 'TV_Horror',
    'Classic_Movies', 'Anime_Features', 'Stand_Up_Comedy_and', 'Crime_TV_Shows',
    'TV_Sci_Fi_and_Fantasy', 'Faith_and_Spiritua'
]


# COMMAND ----------

# DBTITLE 1,Unpivot (melt) countries and genres
# Convert wide country columns to long format
country_genres = (
    df.select(*countries, *genres)
    .selectExpr("stack(33, " + ", ".join([f"'{c}', {c}" for c in countries]) + ") as (country, available)",
                *genres)
    .filter(F.col("available") == 1)
)

# Now unpivot genre columns
country_genres = (
    country_genres.selectExpr("country", "stack(" + str(len(genres)) + ", " +
                              ", ".join([f"'{g}', {g}" for g in genres]) +
                              ") as (genre, present)")
    .filter(F.col("present") == 1)
)


# COMMAND ----------

# DBTITLE 1,ount and get top genre for each country
# Count genre occurrences per country
genre_counts = country_genres.groupBy("country", "genre").count()

# Rank genres within each country
window = Window.partitionBy("country").orderBy(F.desc("count"))

# Get top genre (rank = 1)
top_genres = (
    genre_counts.withColumn("rank", F.row_number().over(window))
    .filter(F.col("rank") == 1)
    .select("country", "genre", "count")
)

display(top_genres)


# COMMAND ----------

# DBTITLE 1,Visualization
import matplotlib.pyplot as plt

# Convert to Pandas for easy plotting
top_genres_pd = top_genres.toPandas()

# Sort by count for clear visualization
top_genres_pd = top_genres_pd.sort_values(by="count", ascending=False)

# Plot
plt.figure(figsize=(14, 6))
plt.bar(top_genres_pd["country"], top_genres_pd["count"], color='green', edgecolor='white')
plt.xticks(rotation=75, ha='right')
plt.title("Top Genre Count by Country", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()


# COMMAND ----------

# DBTITLE 1,Content Rating Distribution Analysis
from pyspark.sql import functions as F

# Count titles by rating
rating_distribution = df.groupBy("rating").agg(F.count("*").alias("count"))

# Calculate total for percentage
total_titles = rating_distribution.agg(F.sum("count").alias("total")).collect()[0]["total"]

# Add percentage column (rounded to 3 decimal places)
rating_distribution = rating_distribution.withColumn(
    "percentage",
    F.round((F.col("count") / total_titles) * 100, 3)
)

# Sort results in descending order
rating_distribution = rating_distribution.orderBy(F.desc("count"))

display(rating_distribution)


# COMMAND ----------

# DBTITLE 1,Plot Rating Distribution
import pandas as pd
import matplotlib.pyplot as plt

# Convert Spark DataFrame to Pandas
rating_pd = rating_distribution.toPandas()

# Plot a bar chart
plt.figure(figsize=(10,6))
plt.bar(rating_pd['rating'], rating_pd['percentage'])
plt.xlabel("Rating")
plt.ylabel("Percentage (%)")
plt.title("Distribution of Netflix Titles by Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# COMMAND ----------

# DBTITLE 1,Movie vs TV Show by Country)
from pyspark.sql import functions as F

countries = ['Argentina', 'Australia', 'Belgium', 'Brazil', 'Canada', 'China', 'Denmark', 'Egypt', 'Finland', 
              'France', 'Germany', 'India', 'Indonesia', 'Italy', 'Japan', 'Mexico', 'Netherlands', 'Nigeria', 
              'Norway', 'Pakistan', 'Philippines', 'Poland', 'Russia', 'South_Africa', 'Spain', 'Sweden', 
              'Switzerland', 'Thailand', 'Turkey', 'USA']

country_stats = []
for country in countries:
    stats = (
        df.filter(F.col(country) == 1)
          .agg(
              F.lit(country).alias("country"),
              F.sum("movie").alias("movie_count"),
              F.sum("tv_show").alias("tvshow_count"),
              F.round((F.sum("movie") / (F.sum("movie") + F.sum("tv_show"))) * 100, 1).alias("movie_percentage"),
              F.round((F.sum("tv_show") / (F.sum("movie") + F.sum("tv_show"))) * 100, 1).alias("tvshow_percentage")
          )
    )
    country_stats.append(stats)

final_country_df = country_stats[0]
for c in country_stats[1:]:
    final_country_df = final_country_df.union(c)

display(final_country_df.orderBy(F.desc("movie_count")))


# COMMAND ----------

# DBTITLE 1,Visualization
import matplotlib.pyplot as plt
import numpy as np

# Convert to pandas
country_pd = final_country_df.toPandas()

# Sort by movie count for clarity
country_pd = country_pd.sort_values(by="movie_count", ascending=False)

# Create positions for bars
x = np.arange(len(country_pd["country"]))
width = 0.35  # width of the bars

# Plot setup
plt.figure(figsize=(14, 6))
plt.bar(x - width/2, country_pd["movie_percentage"], width, label="Movies", color="#FF6F61")
plt.bar(x + width/2, country_pd["tvshow_percentage"], width, label="TV Shows", color="#6FA8DC")

# Titles and labels
plt.title("Movies vs TV Shows Percentage by Country", fontsize=16, fontweight="bold")
plt.xlabel("Country", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.xticks(x, country_pd["country"], rotation=70, ha='right')
plt.legend(title="Content Type")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ANALYSIS OF NETFLIX PRODUCTION BEFORE AND AFTER 2015
# MAGIC

# COMMAND ----------

# DBTITLE 1,Years
from pyspark.sql import functions as F

# 1️⃣ Get start and end year
year_summary = df.select(
    F.min("release_year").alias("start_year"),
    F.max("release_year").alias("end_year")
)

# 2️⃣ Group by production period (Before/After 2015)
production_trend = (
    df.withColumn(
        "production_period",
        F.when(F.col("release_year") < 2015, "Before 2015").otherwise("After 2015")
    )
    .groupBy("production_period")
    .agg(F.count("*").alias("total_titles"))
)

# 3️⃣ Combine and show both results
print("🎬 Netflix Dataset Year Range:")
display(year_summary)

print("📈 Production Trend (Before vs After 2015):")
display(production_trend)


# COMMAND ----------

# DBTITLE 1,titles  released between 2012 - 2021
from pyspark.sql import functions as F

# Filter titles released between 2012 and 2021
trend_yearly = (
    df.filter((F.col("release_year") >= 2012) & (F.col("release_year") <= 2021))
      .groupBy("release_year")
      .agg(F.count("*").alias("total_titles"))
      .orderBy("release_year")
)

display(trend_yearly)


# COMMAND ----------

# DBTITLE 1,Visualization
import matplotlib.pyplot as plt

# Convert to pandas for visualization
trend_pd = trend_yearly.toPandas()

plt.figure(figsize=(10,6))
plt.plot(
    trend_pd["release_year"], 
    trend_pd["total_titles"], 
    marker='o', 
    color='#FF4C29', 
    linewidth=3, 
    label="Total Titles"
)
plt.bar(trend_pd["release_year"], trend_pd["total_titles"], color='darkblue', alpha=0.5)

plt.title("Netflix Production Trend (2012–2021)", fontsize=16, fontweight="bold")
plt.xlabel("Release Year", fontsize=12)
plt.ylabel("Number of Titles Released", fontsize=12)
plt.xticks(trend_pd["release_year"])
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt

trend_pd = production_trend.toPandas()

plt.figure(figsize=(7,5))
plt.bar(trend_pd["production_period"], trend_pd["total_titles"], 
        color=["#A3C1AD", "#F67280"], edgecolor="black")

plt.title("Netflix Production Trend: Before vs After 2015", fontsize=16, fontweight="bold")
plt.ylabel("Number of Titles", fontsize=12)
plt.xlabel("Production Period", fontsize=12)
plt.tight_layout()
plt.show()


# COMMAND ----------

# DBTITLE 1,Featire Imp Analysis
# Get feature importances from trained Random Forest model
feature_importances = rf_model.featureImportances.toArray().tolist()

# Get feature column names (same as used in VectorAssembler)
feature_names = feature_cols  # replace with your actual feature column list

# Create DataFrame of features and their importance (rounded to 4 decimals)
rounded_importances = [round(value, 4) for value in feature_importances]

feature_importance_df = spark.createDataFrame(
    zip(feature_names, rounded_importances),
    ["Feature", "Importance"]
)

display(feature_importance_df)


# COMMAND ----------

# DBTITLE 1,HeatMap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert Spark DataFrame to Pandas for plotting
pandas_df = feature_importance_df.toPandas()

# Sort by importance
pandas_df = pandas_df.sort_values(by="Importance", ascending=True)

# Set up the figure
plt.figure(figsize=(8, len(pandas_df) * 0.4))  # auto height based on number of features

# Create vertical heatmap
sns.heatmap(
    pandas_df[["Importance"]], 
    yticklabels=pandas_df["Feature"],
    annot=True,
    cmap="YlOrRd",
    fmt=".4f",
    cbar_kws={'label': 'Feature Importance'}
)

plt.title("Random Forest Feature Importance Heatmap", fontsize=14, pad=15)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()
