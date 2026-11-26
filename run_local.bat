@echo off
set DM_BACKEND=local
set DM_MODEL=dm-lora
uvicorn app.api:app --port 8000
