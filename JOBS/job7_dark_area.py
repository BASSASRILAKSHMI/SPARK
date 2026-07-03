"""
RDHEI Feature Job 7 - Dark Area Ratio
=======================================
Dark areas are pixels with intensity < 50 (out of 255).
In RDHEI, embedding in very dark regions risks pixel underflow (values going below 0),
which also causes irreversible distortion. Lower dark ratio = safer for embedding.
"""

import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, when

spark = SparkSession.builder \
    .appName("RDHEI-Feature7-DarkAreaRatio") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "512m") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("=" * 60)
print("  FEATURE 7 : Dark Area Ratio")
print("  % of pixels with intensity < 50")
print("  Lower dark ratio = safer for RDHEI (avoids underflow)")
print("=" * 60)

DATASET_PATH = "/opt/spark/dataset"
OUTPUT_PATH  = "/opt/spark/output/feature7_dark_area"

image_paths = [
    os.path.join(DATASET_PATH, f)
    for f in os.listdir(DATASET_PATH) if f.endswith(".png")
]

def compute_dark_area(path):
    import cv2, numpy as np
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (os.path.basename(path), 0.0)
    dark_ratio = float(np.sum(img < 50) / img.size)
    return (os.path.basename(path), dark_ratio)

rdd     = sc.parallelize(image_paths, numSlices=4)
results = rdd.map(compute_dark_area).collect()

df = spark.createDataFrame(results, ["Image", "DarkAreaRatio"])

max_dark = df.agg(spark_max("DarkAreaRatio")).collect()[0][0]
df = df.withColumn("DarkArea_norm", col("DarkAreaRatio") / max_dark)
df = df.withColumn("Label", when(col("DarkArea_norm") <= 0.5, "Suitable").otherwise("Not Suitable"))
df = df.orderBy(col("DarkAreaRatio"))

print("\n[RESULT] Dark Area Ratio per image (lower = safer for RDHEI):")
df.show(20)

print("\n[INFO] Label Distribution:")
df.groupBy("Label").count().show()

df.coalesce(1).write.mode("overwrite").option("header", "true").csv(OUTPUT_PATH)
print(f"[INFO] Saved to {OUTPUT_PATH}")
spark.stop()
