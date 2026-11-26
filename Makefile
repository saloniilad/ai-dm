.RECIPEPREFIX = >
.PHONY: install test eval api cli selfplay sft train docker

install:
> pip install -r requirements-dev.txt

test:
> python -m pytest -q

eval:
> python -m eval.retrieval_eval
> python -m eval.ablation
> python -m eval.behavior_eval

api:
> uvicorn app.api:app --reload

cli:
> python -m app.cli

selfplay:
> python -m train.self_play --sessions 50 --turns 8 --out data/dataset/teacher.jsonl

sft:
> python -m train.build_sft data/dataset/*.jsonl --out data/sft

train:
> python -m train.train_lora --data data/sft --out out/dm-lora

docker:
> docker build -t ai-dm . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY -v dm-data:/data ai-dm
