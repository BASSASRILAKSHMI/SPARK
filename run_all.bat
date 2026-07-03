@echo off
echo ============================================
echo   BDA-RDHEI : Running All 7 Feature Jobs
echo ============================================

echo.
echo [SETUP] Starting Spark Cluster...
docker-compose up -d
timeout /t 15 /nobreak >nul

echo.
echo [SETUP] Installing opencv on all nodes...
docker exec -u root spark-master pip install opencv-python-headless --quiet
docker exec -u root spark-worker-1 pip install opencv-python-headless --quiet
docker exec -u root spark-worker-2 pip install opencv-python-headless --quiet

echo.
echo ============================================
echo  JOB 1 of 7 : Shannon Entropy
echo ============================================
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --deploy-mode client --executor-memory 512m --executor-cores 1 /opt/spark/jobs/job1_entropy.py
echo [DONE] Job 1 complete.

echo.
echo ============================================
echo  JOB 2 of 7 : Edge Density
echo ============================================
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --deploy-mode client --executor-memory 512m --executor-cores 1 /opt/spark/jobs/job2_edge_density.py
echo [DONE] Job 2 complete.

echo.
echo ============================================
echo  JOB 3 of 7 : Pixel Variance
echo ============================================
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --deploy-mode client --executor-memory 512m --executor-cores 1 /opt/spark/jobs/job3_variance.py
echo [DONE] Job 3 complete.

echo.
echo ============================================
echo  JOB 4 of 7 : Smooth Area Ratio
echo ============================================
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --deploy-mode client --executor-memory 512m --executor-cores 1 /opt/spark/jobs/job4_smooth_area.py
echo [DONE] Job 4 complete.

echo.
echo ============================================
echo  JOB 5 of 7 : Embedding Capacity
echo ============================================
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --deploy-mode client --executor-memory 512m --executor-cores 1 /opt/spark/jobs/job5_embedding_capacity.py
echo [DONE] Job 5 complete.

echo.
echo ============================================
echo  JOB 6 of 7 : Bright Area Ratio
echo ============================================
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --deploy-mode client --executor-memory 512m --executor-cores 1 /opt/spark/jobs/job6_bright_area.py
echo [DONE] Job 6 complete.

echo.
echo ============================================
echo  JOB 7 of 7 : Dark Area Ratio
echo ============================================
docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --deploy-mode client --executor-memory 512m --executor-cores 1 /opt/spark/jobs/job7_dark_area.py
echo [DONE] Job 7 complete.

echo.
echo ============================================
echo   ALL 7 JOBS COMPLETED!
echo   Check output\ folder for all CSV results
echo ============================================
pause
