"""
RDHEI Feature Job 2 - Edge Density
=====================================
Edge density measures how many sharp edges exist in the image.
Low edge density = less structural complexity = better PSNR after embedding.
"""

import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, when

spark = SparkSession.builder \
    .appName("RDHEI-Feature2-EdgeDensity") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "512m") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("=" * 60)
print("  FEATURE 2 : Edge Density (Canny Edge Detection)")
print("  Low edge density = more suitable for RDHEI embedding")
print("=" * 60)

DATASET_PATH = "/opt/spark/dataset"
OUTPUT_PATH  = "/opt/spark/output/feature2_edge_density"

image_paths = [
    os.path.join(DATASET_PATH, f)
    for f in os.listdir(DATASET_PATH) if f.endswith(".png")
]

def compute_edge_density(path):
    import cv2, numpy as np
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (os.path.basename(path), 0.0)
    edges = cv2.Canny(img, 100, 200)
    edge_density = float(np.sum(edges > 0) / img.size)
    return (os.path.basename(path), edge_density)

rdd     = sc.parallelize(image_paths, numSlices=4)
results = rdd.map(compute_edge_density).collect()

df = spark.createDataFrame(results, ["Image", "EdgeDensity"])

max_edge = df.agg(spark_max("EdgeDensity")).collect()[0][0]
df = df.withColumn("EdgeDensity_norm", col("EdgeDensity") / max_edge)
df = df.withColumn("Label", when(col("EdgeDensity_norm") <= 0.6, "Suitable").otherwise("Not Suitable"))
df = df.orderBy(col("EdgeDensity"))

print("\n[RESULT] Edge Density per image (lower = better for RDHEI):")
df.show(20)

print("\n[INFO] Label Distribution:")
df.groupBy("Label").count().show()

df.coalesce(1).write.mode("overwrite").option("header", "true").csv(OUTPUT_PATH)
print(f"[INFO] Saved to {OUTPUT_PATH}")
spark.stop()
