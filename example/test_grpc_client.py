import grpc
import json
import sys

try:
    import firefly_pb2
    import firefly_pb2_grpc
except ImportError:
    print("Please generate python grpc stubs first: python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. ../proto/firefly.proto")
    sys.exit(1)

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = firefly_pb2_grpc.InferenceServiceStub(channel)
        
        request = firefly_pb2.ChatCompletionRequest(
            model="qwen3",
            max_tokens=2048
        )
        msg = request.messages.add()
        msg.role = "user"
        msg.content = "What is the capital of France?"
        
        print("Sending request...")
        response = stub.ChatCompletion(request)
        
        print("\n=== Response ===")
        print(f"ID: {response.id}")
        for choice in response.choices:
            if choice.reasoning_content:
                print(f"[Thinking]: {choice.reasoning_content}")
            print(f"[Content]: {choice.message.content}")

if __name__ == '__main__':
    run()
