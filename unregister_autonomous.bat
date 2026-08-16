@echo off
set STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\StartUp
del /F "%STARTUP%\memagent_autonomous.vbs" 2>nul
echo [OK] unregistered.
pause
