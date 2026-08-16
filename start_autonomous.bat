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
rem 2026-08-16 起改为新书《违约金》：
rem   --store novel_memory.json      独立记忆库（旧书《错季锁星》的设定不污染新书人设档案）
rem   --persona-file works\personas\违约金.txt   读者口味人设（杀伐果断+规则解谜必兑付+都市底色）
rem   --no-evolve                    大纲驱动：关闭每轮新设定注入（防谜题通胀，旧书根因）
"%PYTHON%" autonomous_writer.py --store novel_memory.json --persona-file "works\personas\违约金.txt" --no-evolve --cycles 10
