@echo off
set "APPDIR=%~dp0"
set "PYTHON=C:\Python311\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
cd /d "%APPDIR%"
if exist .env (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
    set "%%a=%%b"
  )
)
"%PYTHON%" autonomous_writer.py --persona novelist --cycles 10
