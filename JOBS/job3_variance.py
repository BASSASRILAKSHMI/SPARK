"""
RDHEI Feature Job 3 - Pixel Variance
======================================
Variance measures how spread out pixel intensities are.
Low variance = smoother pixel distribution = better reversibility after extraction.
"""

import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, when

spark = SparkSession.builder \
    .appName("RDHEI-Feature3-Variance") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "512m") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("=" * 60)
print("  FEATURE 3 : Pixel Variance")
print("  Low variance = more suitable for RDHEI embedding")
print("=" * 60)

DATASET_PATH = "/opt/spark/dataset"
OUTPUT_PATH  = "/opt/spark/output/feature3_variance"

image_paths = [
    os.path.join(DATASET_PATH, f)
    for f in os.listdir(DATASET_PATH) if f.endswith(".png")
]

def compute_variance(path):
    import cv2, numpy as np
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (os.path.basename(path), 0.0)
    variance = float(np.var(img))
    return (os.path.basename(path), variance)

rdd     = sc.parallelize(image_paths, numSlices=4)
results = rdd.map(compute_variance).collect()

df = spark.createDataFrame(results, ["Image", "Variance"])

max_var = df.agg(spark_max("Variance")).collect()[0][0]
df = df.withColumn("Variance_norm", col("Variance") / max_var)
df = df.withColumn("Label", when(col("Variance_norm") <= 0.6, "Suitable").otherwise("Not Suitable"))
df = df.orderBy(col("Variance"))

print("\n[RESULT] Pixel Variance per image (lower = better for RDHEI):")
df.show(20)

print("\n[INFO] Label Distribution:")
df.groupBy("Label").count().show()

df.coalesce(1).write.mode("overwrite").option("header", "true").csv(OUTPUT_PATH)
print(f"[INFO] Saved to {OUTPUT_PATH}")
spark.stop()
