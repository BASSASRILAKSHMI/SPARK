@echo off
echo ============================================
echo   BDA-RDHEI Spark Cluster - Auto Runner
echo ============================================

echo.
echo [1/5] Starting Spark Cluster...
docker-compose up -d
timeout /t 10 /nobreak >nul

echo.
echo [2/5] Installing opencv on Master...
docker exec -u root spark-master pip install opencv-python-headless --quiet

echo.
echo [3/5] Installing opencv on Worker-1...
docker exec -u root spark-worker-1 pip install opencv-python-headless --quiet

echo.
echo [4/5] Installing opencv on Worker-2...
docker exec -u root spark-worker-2 pip install opencv-python-headless --quiet

echo.
echo [5/5] Submitting PySpark Job to Cluster...
docker exec spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --deploy-mode client ^
  --executor-memory 512m ^
  --executor-cores 1 ^
  /opt/spark/jobs/rdhei_spark_job.py

echo.
echo ============================================
echo   Done! Check output/ folder for CSV
echo ============================================
pause