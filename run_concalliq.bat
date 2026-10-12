@echo off
title ConcallIQ Earnings Intelligence
cd /d "%~dp0"
echo =====================================================================
echo Starting ConcallIQ - Institutional Earnings Intelligence Platform
echo =====================================================================
python -m streamlit run app.py
pause
