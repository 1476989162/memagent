@echo off
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\StartUp"
set "RUNNER=%~dp0start_autonomous.bat"
set "TARGET=%STARTUP%\memagent_autonomous.vbs"
if not exist "%RUNNER%" (
  echo [ERROR] missing: %RUNNER%
  pause
  exit /b 1
)
if not exist "%STARTUP%" mkdir "%STARTUP%"
> "%TARGET%" echo Set WshShell = CreateObject("WScript.Shell")
>> "%TARGET%" echo WshShell.Run "cmd.exe /c ""%RUNNER%""", 0, False
echo [OK] registered. Runner: %TARGET%
echo To uninstall: run unregister_autonomous.bat or delete the .vbs above.
pause
