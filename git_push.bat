@echo off
cd /d "C:\Users\ASUS\Downloads\anticheating"
echo Initializing Git repository...
git init
echo.
echo Adding all files...
git add .
echo.
echo Committing files...
git commit -m "Anti-cheating AI system - initial commit"
echo.
echo Adding remote origin...
git remote add origin https://github.com/shivamshrma09/cheatingdeetction.git
echo.
echo Setting main branch...
git branch -M main
echo.
echo Pushing to GitHub...
git push -u origin main
echo.
echo Push complete! Repository updated.
pause