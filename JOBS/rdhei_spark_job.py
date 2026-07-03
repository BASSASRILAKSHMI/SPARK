import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, expr, when

spark = SparkSession.builder \
    .appName("RDHEI-ImageSuitability-ClusterMode") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "512m") \
    .config("spark.executor.cores", "1") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("=" * 60)
print("  Spark Master URL :", sc.master)
print("  App Name         :", sc.appName)
print("=" * 60)

DATASET_PATH = "/opt/spark/dataset"
OUTPUT_PATH  = "/opt/spark/output"

image_paths = [
    os.path.join(DATASET_PATH, fname)
    for fname in os.listdir(DATASET_PATH)
    if fname.endswith(".png")
]

if not image_paths:
    print("ERROR: No images found in", DATASET_PATH)
    sys.exit(1)

print(f"\n[INFO] Found {len(image_paths)} images - distributing across workers...\n")

def extract_features(path):
    import cv2
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (os.path.basename(path), 0.0, 0.0, 0.0)

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    prob = hist / hist.sum()
    entropy = float(-np.sum(prob * np.log2(prob + 1e-7)))

    edges = cv2.Canny(img, 100, 200)
    edge_density = float(np.sum(edges > 0) / img.size)

    variance = float(np.var(img))

    return (os.path.basename(path), entropy, edge_density, variance)

rdd = sc.parallelize(image_paths, numSlices=4)
results = rdd.map(extract_features).collect()

print(f"[INFO] Feature extraction complete. {len(results)} images processed.\n")

df = spark.createDataFrame(results, ["Image", "Entropy", "EdgeDensity", "Variance"])
df.show(10)

max_entropy  = df.agg(spark_max("Entropy")).collect()[0][0]
max_edge     = df.agg(spark_max("EdgeDensity")).collect()[0][0]
max_variance = df.agg(spark_max("Variance")).collect()[0][0]

df_norm = df \
    .withColumn("Entropy_norm", col("Entropy")    / max_entropy) \
    .withColumn("Edge_norm",    col("EdgeDensity") / max_edge) \
    .withColumn("Var_norm",     col("Variance")    / max_variance)

df_score = df_norm.withColumn(
    "Score",
    expr("(1 - Entropy_norm) + (1 - Edge_norm) + (1 - Var_norm)")
)

df_final = df_score.orderBy(col("Score").desc())

total = df_final.count()
top_n = int(0.4 * total)
threshold = df_final.limit(top_n).agg({"Score": "min"}).collect()[0][0]

df_labeled = df_final.withColumn(
    "Label",
    when(col("Score") >= threshold, "Suitable").otherwise("Not Suitable")
)

print("=" * 60)
print("  RDHEI SUITABILITY RESULTS")
print("=" * 60)
df_labeled.select("Image", "Entropy", "EdgeDensity", "Variance", "Score", "Label").show(20)

print("\n[INFO] Label distribution:")
df_labeled.groupBy("Label").count().show()

output_csv = os.path.join(OUTPUT_PATH, "rdhei_results")
df_labeled.select("Image", "Entropy", "EdgeDensity", "Variance", "Score", "Label") \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(output_csv)

print(f"\n[INFO] Results saved to {output_csv}")
print("\n[DONE] Job completed successfully on Spark Cluster.")
spark.stop()