"""
RDHEI Feature Job 1 - Shannon Entropy
======================================
Entropy measures the randomness/complexity of pixel intensities.
Low entropy = uniform image = easier to embed hidden data reversibly.
"""

import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, when

spark = SparkSession.builder \
    .appName("RDHEI-Feature1-Entropy") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "512m") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("=" * 60)
print("  FEATURE 1 : Shannon Entropy")
print("  Low entropy = more suitable for RDHEI embedding")
print("=" * 60)

DATASET_PATH = "/opt/spark/dataset"
OUTPUT_PATH  = "/opt/spark/output/feature1_entropy"

image_paths = [
    os.path.join(DATASET_PATH, f)
    for f in os.listdir(DATASET_PATH) if f.endswith(".png")
]

def compute_entropy(path):
    import cv2, numpy as np
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (os.path.basename(path), 0.0)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    prob = hist / hist.sum()
    entropy = float(-np.sum(prob * np.log2(prob + 1e-7)))
    return (os.path.basename(path), entropy)

rdd     = sc.parallelize(image_paths, numSlices=4)
results = rdd.map(compute_entropy).collect()

df = spark.createDataFrame(results, ["Image", "Entropy"])

max_entropy = df.agg(spark_max("Entropy")).collect()[0][0]
df = df.withColumn("Entropy_norm", col("Entropy") / max_entropy)
df = df.withColumn("Label", when(col("Entropy_norm") <= 0.6, "Suitable").otherwise("Not Suitable"))
df = df.orderBy(col("Entropy"))

print("\n[RESULT] Entropy per image (lower = better for RDHEI):")
df.show(20)

print("\n[INFO] Label Distribution:")
df.groupBy("Label").count().show()

df.coalesce(1).write.mode("overwrite").option("header", "true").csv(OUTPUT_PATH)
print(f"[INFO] Saved to {OUTPUT_PATH}")
spark.stop()
