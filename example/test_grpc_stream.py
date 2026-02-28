import grpc
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
            max_tokens=4096
        )
        msg = request.messages.add()
        msg.role = "user"
        msg.content = "Write a python script to calculate the fibonacci series, please show your reasoning step by step."
        
        print("Sending streaming request...\n")
        
        try:
            responses = stub.ChatCompletionStream(request)
            for response in responses:
                for choice in response.choices:
                    if choice.delta.reasoning_content:
                        sys.stdout.write("\033[90m" + choice.delta.reasoning_content + "\033[0m")
                        sys.stdout.flush()
                    if choice.delta.content:
                        sys.stdout.write(choice.delta.content)
                        sys.stdout.flush()
            print()
        except grpc.RpcError as e:
            print(f"\nRPC failed: {e.code()} - {e.details()}")

if __name__ == '__main__':
    run()
