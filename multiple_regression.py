# Michelle La
# May 9, 2019

# Data obtained from:
# Cuffey, K. M. et al. (2007)
# "Ablation Rates of Taylor Glacier, Antarctica" U.S. Antarctic Program (USAP) Data Center. doi: 10.7265/N5N29TW8.

# Adapted from source:
# https://towardsdatascience.com/building-a-linear-regression-with-pyspark-and-mllib-d065c3ba246a


from __future__ import division

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import PolynomialExpansion

# For plotting purposes only
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


file_list = ['Ant4_USAP-DC_AblationTaylor/TaylorAblationData03_04.txt',
             'Ant4_USAP-DC_AblationTaylor/TaylorAblationData03_06.txt',
             'Ant4_USAP-DC_AblationTaylor/TaylorAblationData03_07.txt',
             'Ant4_USAP-DC_AblationTaylor/TaylorAblationData03_10.txt',
             'Ant4_USAP-DC_AblationTaylor/TaylorAblationData03_11.txt',
             'Ant4_USAP-DC_AblationTaylor/TaylorAblationData04_L4.txt']

spark = SparkSession.builder.appName("Python_Spark").getOrCreate()


for idx, file in enumerate(file_list):  # For first file, "initialize" DataFrame; subsequent ones, append
    if idx == 0:
        data = spark.read.text(file).withColumn("elevation", F.split("value", '\s')[5].cast(FloatType()))\
                                    .withColumn("end_year", F.split("value", '\s')[9].cast(FloatType())) \
                                    .withColumn("end_month", F.split("value", '\s')[10].cast(FloatType())) \
                                    .withColumn("ablation_rate", F.split("value", '\s')[12].cast(FloatType()))\
                                    .drop("value")
    # .withColumn("ablation_pole_name", F.split("value", '\s')[0]) \
    # .withColumn("latitude", F.split("value", '\s')[1]) \
    # .withColumn("longitude", F.split("value", '\s')[2]) \
    # .withColumn("northing", F.split("value", '\s')[3]) \
    # .withColumn("easting", F.split("value", '\s')[4]) \
    # .withColumn("start_year", F.split("value", '\s')[6]) \
    # .withColumn("start_month", F.split("value", '\s')[7]) \
    # .withColumn("start_day", F.split("value", '\s')[8]) \
    # .withColumn("end_day", F.split("value", '\s')[11]) \
    else:
        data_temp = spark.read.text(file).withColumn("elevation", F.split("value", '\s')[5].cast(FloatType())) \
                                         .withColumn("end_year", F.split("value", '\s')[9].cast(FloatType())) \
                                         .withColumn("end_month", F.split("value", '\s')[10].cast(FloatType())) \
                                         .withColumn("ablation_rate", F.split("value", '\s')[12].cast(FloatType())) \
                                         .drop("value")
        data = data.unionAll(data_temp)

data = data.withColumn("time", ((data["end_month"] / 12) + data["end_year"]).cast(FloatType())) \
       .drop("end_year", "end_month")
print("Data count")
print(data.count())

# Get min/max values
min_value = data.agg(F.min("elevation")).collect()[0][0]
max_value = data.agg(F.max("elevation")).collect()[0][0]
print("Min/max elevation: " + str(min_value) + " and " + str(max_value))
min_value = data.agg(F.min("ablation_rate")).collect()[0][0]
max_value = data.agg(F.max("ablation_rate")).collect()[0][0]
print("Min/max ablation rate: " + str(min_value) + " and " + str(max_value))

# Transform independent variable columns into vector of features
vectorAssembler = VectorAssembler(inputCols=["elevation", "time"], outputCol="features")
vector_data = vectorAssembler.transform(data)
vector_data = vector_data.select(["features", "ablation_rate"])
vector_data.show(vector_data.count(), truncate=False)

# Convert to polynomial features
polyExpansion = PolynomialExpansion(degree=1, inputCol='features', outputCol='polyFeatures')
poly_data = polyExpansion.transform(vector_data)
poly_data = poly_data.select(["polyFeatures", "ablation_rate"])
poly_data.show(truncate=False)

# Split into training and test data sets
splits = poly_data.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]
print("Train data count")
print(train_df.count())
print("Test data count")
print(test_df.count())

lr = LinearRegression(featuresCol='polyFeatures', labelCol='ablation_rate', regParam=0.01)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
train_df.describe().show()

predictions = lr_model.transform(test_df)
predictions.select("prediction","ablation_rate","polyFeatures").show()


# Convert to pandas DataFrame for plotting (Spark has no plotting capabilities)
pd_predictions = predictions.toPandas()
pd_predictions['elevation'] = pd_predictions['polyFeatures'].str[0]
pd_predictions['time'] = pd_predictions['polyFeatures'].str[1]

pd_predictions_plot = pd_predictions.plot.scatter(x='ablation_rate', y='prediction', c='elevation', colormap='jet',
                                                  title='Ablation Rate Linear Regression Model Predictions')
pd_predictions_plot.set_xlabel('Actual Value')
pd_predictions_plot.set_xlim(0,0.5)
pd_predictions_plot.set_ylabel('Predicted Value')
pd_predictions_plot.set_ylim(0,0.5)
plt.show()

time = [2020, 2050, 2100, 2200]
# y = 20.83185446931431 - 0.000197696(elevation) - 0.0102181(time)
# elevation = 400, 1500 (based loosely off the general range excluding outliers of the original values)
year_2020 = [[400, 1500], [0.11221, -0.10525]]
year_2050 = [[400, 1500], [-0.19433, -0.41179]]
year_2100 = [[400, 1500], [-0.70523, -0.9227]]
year_2200 = [[400, 1500], [-1.72704, -1.94451]]

line_plot = plt.plot(year_2020[0], year_2020[1], year_2050[0], year_2050[1], year_2100[0], year_2100[1], year_2200[0], year_2200[1], marker='o')
plt.legend(['2020', '2050', '2100', '2200'], loc=1)
plt.xlim(300, 2000)
plt.xlabel('Elevation')
plt.ylabel('Ablation Rate')
plt.title('Ablation Rate Linear Regression Model Predictions')
plt.show()