"""
RDHEI Feature Job 5 - Embedding Capacity
==========================================
Embedding Capacity estimates how many bits of secret data an image can carry.
Based on the number of smooth pixels — each smooth pixel can hold 1 bit (LSB-based RDHEI).
Higher capacity = more data can be hidden = more useful image for RDHEI.
"""

import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, when

spark = SparkSession.builder \
    .appName("RDHEI-Feature5-EmbeddingCapacity") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "512m") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("=" * 60)
print("  FEATURE 5 : Embedding Capacity (estimated bits)")
print("  Higher capacity = more bits can be hidden in this image")
print("=" * 60)

DATASET_PATH = "/opt/spark/dataset"
OUTPUT_PATH  = "/opt/spark/output/feature5_embedding_capacity"

image_paths = [
    os.path.join(DATASET_PATH, f)
    for f in os.listdir(DATASET_PATH) if f.endswith(".png")
]

def compute_embedding_capacity(path):
    import cv2, numpy as np
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (os.path.basename(path), 0.0, 0)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    smooth_pixels = int(np.sum(gradient_mag < 10))
    capacity_bits = smooth_pixels          # 1 bit per smooth pixel
    capacity_bytes = smooth_pixels // 8    # convert to bytes
    return (os.path.basename(path), float(capacity_bits), float(capacity_bytes))

rdd     = sc.parallelize(image_paths, numSlices=4)
results = rdd.map(compute_embedding_capacity).collect()

df = spark.createDataFrame(results, ["Image", "CapacityBits", "CapacityBytes"])

max_bits = df.agg(spark_max("CapacityBits")).collect()[0][0]
df = df.withColumn("Capacity_norm", col("CapacityBits") / max_bits)
df = df.withColumn("Label", when(col("Capacity_norm") >= 0.4, "Suitable").otherwise("Not Suitable"))
df = df.orderBy(col("CapacityBits").desc())

print("\n[RESULT] Embedding Capacity per image (higher = better for RDHEI):")
df.show(20)

print("\n[INFO] Label Distribution:")
df.groupBy("Label").count().show()

df.coalesce(1).write.mode("overwrite").option("header", "true").csv(OUTPUT_PATH)
print(f"[INFO] Saved to {OUTPUT_PATH}")
spark.stop()
