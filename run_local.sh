#!/usr/bin/env bash
DM_BACKEND=local DM_MODEL=dm-lora uvicorn app.api:app --port 8000
