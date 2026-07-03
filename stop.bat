@echo off
echo ============================================
echo   BDA-RDHEI Spark Cluster - Stopping...
echo ============================================

echo.
echo [1/2] Stopping all containers...
docker-compose down

echo.
echo [2/2] Cluster stopped successfully!
echo       Your data in dataset/ and output/ is safe.
echo.
echo ============================================
echo   Goodbye!
echo ============================================
pause