@echo off
cd /d "C:\Users\ASUS\Downloads\anticheating"
echo Current directory: %CD%
echo.
echo Step 1: Installing Railway CLI...
call npm install -g @railway/cli
echo.
echo Step 2: Login to Railway (browser will open)...
call railway login
echo.
echo Step 3: Initialize project...
call railway init
echo.
echo Step 4: Deploying...
call railway up
echo.
echo Deployment complete! Copy the URL from above.
pause