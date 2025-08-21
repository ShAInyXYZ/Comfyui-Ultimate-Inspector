@echo off
cd ..
set datetime=%date:~10,4%-%date:~4,2%-%date:~7,2%_%time:~0,2%-%time:~3,2%
set datetime=%datetime: =0%
python ComfyUI-Ultimate-Inspector\ComfyUI-Ultimate-Inspector.py --output report-%datetime%.md --verbose
pause