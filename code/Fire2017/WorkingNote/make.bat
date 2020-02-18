@echo off
for /r %%i in (rightstepspune*author*.tex) do texify -cp %%i
