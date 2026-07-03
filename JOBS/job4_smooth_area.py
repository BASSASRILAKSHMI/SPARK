"""
RDHEI Feature Job 4 - Smooth Area Ratio
=========================================
Smooth areas are regions where neighbouring pixels differ very little (low gradient).
These are the BEST regions for reversible data hiding — embedding here causes
minimal distortion and allows perfect recovery of original pixel values.
"""

import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, when

spark = SparkSession.builder \
    .appName("RDHEI-Feature4-SmoothAreaRatio") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "512m") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("=" * 60)
print("  FEATURE 4 : Smooth Area Ratio")
print("  High smooth ratio = more regions available for RDHEI embedding")
print("=" * 60)

DATASET_PATH = "/opt/spark/dataset"
OUTPUT_PATH  = "/opt/spark/output/feature4_smooth_area"

image_paths = [
    os.path.join(DATASET_PATH, f)
    for f in os.listdir(DATASET_PATH) if f.endswith(".png")
]

def compute_smooth_area(path):
    import cv2, numpy as np
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (os.path.basename(path), 0.0)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    smooth_ratio = float(np.sum(gradient_mag < 10) / img.size)
    return (os.path.basename(path), smooth_ratio)

rdd     = sc.parallelize(image_paths, numSlices=4)
results = rdd.map(compute_smooth_area).collect()

df = spark.createDataFrame(results, ["Image", "SmoothAreaRatio"])

max_smooth = df.agg(spark_max("SmoothAreaRatio")).collect()[0][0]
df = df.withColumn("SmoothArea_norm", col("SmoothAreaRatio") / max_smooth)
df = df.withColumn("Label", when(col("SmoothArea_norm") >= 0.4, "Suitable").otherwise("Not Suitable"))
df = df.orderBy(col("SmoothAreaRatio").desc())

print("\n[RESULT] Smooth Area Ratio per image (higher = better for RDHEI):")
df.show(20)

print("\n[INFO] Label Distribution:")
df.groupBy("Label").count().show()

df.coalesce(1).write.mode("overwrite").option("header", "true").csv(OUTPUT_PATH)
print(f"[INFO] Saved to {OUTPUT_PATH}")
spark.stop()
