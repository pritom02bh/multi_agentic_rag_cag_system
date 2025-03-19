@echo off
echo Starting Pharmaceutical Supply Chain Management System

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install Python 3.8 or later.
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment. Please check your Python installation.
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    exit /b 1
)

REM Install or upgrade dependencies
echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies.
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo No .env file found. Creating template .env file...
    (
        echo # API Keys
        echo OPENAI_API_KEY=your_openai_api_key
        echo PINECONE_API_KEY=your_pinecone_api_key
        echo PINECONE_ENVIRONMENT=your_pinecone_environment
        echo PINECONE_INDEX=medical-supply-chain
        echo NEWS_API_KEY=your_news_api_key
        echo
        echo # Server Configuration
        echo HOST=0.0.0.0
        echo PORT=5000
        echo DEBUG=True
        echo
        echo # Redis Configuration (optional)
        echo REDIS_URL=redis://localhost:6379/0
    ) > .env
    echo Please edit the .env file with your API keys before running the application.
    exit /b 1
)

REM Run the application
echo Starting application...
python run.py

REM Deactivate virtual environment when done
call venv\Scripts\deactivate.bat 