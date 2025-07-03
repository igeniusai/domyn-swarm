#!/bin/bash

export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
export URL="http://lrdn3271:9000"

for CONCURRENCY_LEVEL in 1 10 100 1000; 
    do genai-perf profile \
        -m $MODEL_NAME \
        --endpoint-type chat \
        --num-prompts 100 \
        --random-seed 123 \
        --streaming \
        --synthetic-input-tokens-mean 1000 \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean 1000 \
        --output-tokens-stddev 0 \
        --concurrency ${CONCURRENCY_LEVEL} \
        --measurement-interval 60000 \
        --warmup-request-count 10 \
        --profile-export-file my_profile_export.json \
        --url $URL; 
done

http://lrdn3375:9000

python vllm_benchmarks/benchmarks/benchmark_serving.py --base-url http://lrdn3381:9001 --max-concurrency 100 --dataset-name sonnet --dataset-path vllm_benchmarks/benchmarks/sonnet.txt --model deepseek-ai/DeepSeek-R1-0528

python vllm_benchmarks/benchmarks/benchmark_serving.py --base-url http://lrdn3345:9000 --max-concurrency 100 --dataset-name sonnet --dataset-path vllm_benchmarks/benchmarks/sonnet.txt --model Qwen/Qwen3-235B-A22B