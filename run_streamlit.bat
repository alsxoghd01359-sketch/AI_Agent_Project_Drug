@echo off
REM 더블클릭하거나 터미널에서 run_streamlit 이라고 치면 Streamlit 채팅 UI가 열립니다.
REM conda 환경 활성화 여부와 무관하게 항상 동작합니다.
cd /d "%~dp0"
"C:\Users\201\anaconda3\envs\ai_agent_challenge\python.exe" -m streamlit run src\streamlit_app.py
pause
