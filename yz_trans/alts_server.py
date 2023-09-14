# Copyright 2020 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The example of using ALTS credentials to setup gRPC server in python.

The example would only successfully run in GCP environment."""

from concurrent import futures

import grpc

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "yz_trans")))

import demo_pb2_grpc
from server import DemoServer

SERVER_ADDRESS = "0.0.0.0:50051"

def build_svrc(addr):
    svr = grpc.server(futures.ThreadPoolExecutor())
    svrc = DemoServer()
    demo_pb2_grpc.add_GRPCDemoServicer_to_server(svrc, svr)
    svr.add_insecure_port(addr)
    print("------------------start Python GRPC server------------------")
    svr.start()
    return svr, svrc
    
def main():
    opts = [
        ("grpc.max_send_message_length", 1000 * 1024 * 1024),
        ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
        ("grpc.enable_http_proxy", 0),
    ]
    svr = grpc.server(futures.ThreadPoolExecutor(), options=opts)
    demo_pb2_grpc.add_GRPCDemoServicer_to_server(DemoServer(), svr)
    # svr.add_secure_port(
    #     SERVER_ADDRESS, server_credentials=grpc.alts_server_credentials()
    # )
    svr.add_insecure_port(SERVER_ADDRESS)
    print("------------------start Python GRPC server with ALTS encryption")
    svr.start()
    svr.wait_for_termination()


if __name__ == "__main__":
    main()
