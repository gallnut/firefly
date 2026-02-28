import asyncio
import time
import argparse
import grpc
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../example'))

try:
    import firefly_pb2
    import firefly_pb2_grpc
except ImportError:
    print("Please generate python grpc stubs first.")
    sys.exit(1)

async def run_single_request(stub, request, req_id):
    start_time = time.time()
    num_tokens = 0
    first_token_time = None
    
    try:
        async for response in stub.ChatCompletionStream(request):
            if first_token_time is None:
                first_token_time = time.time()
            
            for choice in response.choices:
                if choice.delta.content or choice.delta.reasoning_content:
                    num_tokens += 1
                    
        end_time = time.time()
        ttft = first_token_time - start_time if first_token_time else 0
        total_time = end_time - start_time
        
        return {
            "id": req_id,
            "success": True,
            "ttft": ttft,
            "total_time": total_time,
            "tokens": num_tokens
        }
    except Exception as e:
        print(f"Request {req_id} failed: {e}")
        return {
            "id": req_id,
            "success": False,
            "ttft": 0,
            "total_time": time.time() - start_time,
            "tokens": 0
        }

async def worker(stub, request_template, queue, results):
    while True:
        req_id = await queue.get()
        if req_id is None:
            break
            
        request = firefly_pb2.ChatCompletionRequest()
        request.CopyFrom(request_template)
        
        result = await run_single_request(stub, request, req_id)
        results.append(result)
        
        queue.task_done()

async def main(args):
    print(f"Starting benchmark with {args.concurrency} concurrent requests, {args.num_requests} total requests.")
    
    request = firefly_pb2.ChatCompletionRequest(
        model=args.model,
        max_tokens=args.max_tokens
    )
    msg = request.messages.add()
    msg.role = "user"
    msg.content = args.prompt
    
    queue = asyncio.Queue()
    results = []
    
    for i in range(args.num_requests):
        queue.put_nowait(i)
        
    for _ in range(args.concurrency):
        queue.put_nowait(None)
        
    start_time = time.time()
    
    async with grpc.aio.insecure_channel(f'localhost:{args.port}') as channel:
        stub = firefly_pb2_grpc.InferenceServiceStub(channel)
        
        tasks = []
        for _ in range(args.concurrency):
            task = asyncio.create_task(worker(stub, request, queue, results))
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    successful_results = [r for r in results if r["success"]]
    failed_requests = len(results) - len(successful_results)
    
    if not successful_results:
        print("All requests failed!")
        return
        
    total_tokens = sum(r["tokens"] for r in successful_results)
    ttfts = [r["ttft"] for r in successful_results if r["ttft"] > 0]
    
    # Calculate TPOT (Time Per Output Token)
    tpots = []
    for r in successful_results:
        if r["tokens"] > 1:
            tpot = (r["total_time"] - r["ttft"]) / (r["tokens"] - 1)
            tpots.append(tpot)
            
    print("\n--- Benchmark Results ---")
    print(f"Total Requests:      {args.num_requests}")
    print(f"Successful Requests: {len(successful_results)}")
    print(f"Failed Requests:     {failed_requests}")
    print(f"Total Time:          {total_time:.2f} s")
    print(f"Total Tokens:        {total_tokens}")
    print(f"Throughput:          {total_tokens / total_time:.2f} tokens/s")
    print(f"Request Throughput:  {len(successful_results) / total_time:.2f} req/s")
    
    if ttfts:
        print("\n--- Time To First Token (TTFT) ---")
        print(f"Mean TTFT:           {np.mean(ttfts):.4f} s")
        print(f"Median TTFT:         {np.median(ttfts):.4f} s")
        print(f"P90 TTFT:            {np.percentile(ttfts, 90):.4f} s")
        print(f"P99 TTFT:            {np.percentile(ttfts, 99):.4f} s")
        
    if tpots:
        print("\n--- Time Per Output Token (TPOT) ---")
        print(f"Mean TPOT:           {np.mean(tpots)*1000:.2f} ms/token")
        print(f"Median TPOT:         {np.median(tpots)*1000:.2f} ms/token")
        print(f"P90 TPOT:            {np.percentile(tpots, 90)*1000:.2f} ms/token")
        print(f"P99 TPOT:            {np.percentile(tpots, 99)*1000:.2f} ms/token")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Firefly Benchmark Tool")
    parser.add_argument("-c", "--concurrency", type=int, default=4, help="Number of concurrent requests")
    parser.add_argument("-n", "--num-requests", type=int, default=20, help="Total number of requests")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--model", type=str, default="qwen3", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--prompt", type=str, default="Write a python script to calculate the fibonacci series.", help="Prompt to send")
    
    args = parser.parse_args()
    asyncio.run(main(args))
