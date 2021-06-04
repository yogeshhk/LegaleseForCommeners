@echo off
for /r %%i in (*.tex) do texify -cp %%i
