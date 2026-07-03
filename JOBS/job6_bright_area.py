"""
RDHEI Feature Job 6 - Bright Area Ratio
=========================================
Bright areas are pixels with intensity > 180 (out of 255).
In RDHEI, embedding in very bright regions risks pixel overflow (values exceeding 255),
which can cause irreversible distortion. Lower bright ratio = safer for embedding.
"""

import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, when

spark = SparkSession.builder \
    .appName("RDHEI-Feature6-BrightAreaRatio") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "512m") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("=" * 60)
print("  FEATURE 6 : Bright Area Ratio")
print("  % of pixels with intensity > 180")
print("  Lower bright ratio = safer for RDHEI (avoids overflow)")
print("=" * 60)

DATASET_PATH = "/opt/spark/dataset"
OUTPUT_PATH  = "/opt/spark/output/feature6_bright_area"

image_paths = [
    os.path.join(DATASET_PATH, f)
    for f in os.listdir(DATASET_PATH) if f.endswith(".png")
]

def compute_bright_area(path):
    import cv2, numpy as np
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (os.path.basename(path), 0.0)
    bright_ratio = float(np.sum(img > 180) / img.size)
    return (os.path.basename(path), bright_ratio)

rdd     = sc.parallelize(image_paths, numSlices=4)
results = rdd.map(compute_bright_area).collect()

df = spark.createDataFrame(results, ["Image", "BrightAreaRatio"])

max_bright = df.agg(spark_max("BrightAreaRatio")).collect()[0][0]
df = df.withColumn("BrightArea_norm", col("BrightAreaRatio") / max_bright)
df = df.withColumn("Label", when(col("BrightArea_norm") <= 0.5, "Suitable").otherwise("Not Suitable"))
df = df.orderBy(col("BrightAreaRatio"))

print("\n[RESULT] Bright Area Ratio per image (lower = safer for RDHEI):")
df.show(20)

print("\n[INFO] Label Distribution:")
df.groupBy("Label").count().show()

df.coalesce(1).write.mode("overwrite").option("header", "true").csv(OUTPUT_PATH)
print(f"[INFO] Saved to {OUTPUT_PATH}")
spark.stop()
