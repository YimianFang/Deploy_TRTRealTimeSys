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
"""The example of using ALTS credentials to setup gRPC client.

The example would only successfully run in GCP environment."""

import grpc

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "yz_trans")))

from client import bidirectional_streaming_method
from client import client_streaming_method
from client import server_streaming_method
from client import simple_method
import demo_pb2_grpc

SERVER_ADDRESS = "0.0.0.0:50051"

def build_stub(addr):
    channel = grpc.insecure_channel(addr)
    stub = demo_pb2_grpc.GRPCDemoStub(channel)
    return stub

def yz_trans(stub, trans):
    simple_method(stub, trans)

def main():
    # with grpc.secure_channel(
    #     SERVER_ADDRESS, credentials=grpc.alts_channel_credentials()
    # ) as channel:
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        stub = demo_pb2_grpc.GRPCDemoStub(channel)
        simple_method(stub, "hhhhhh")
        # client_streaming_method(stub)
        # server_streaming_method(stub)
        # bidirectional_streaming_method(stub)


if __name__ == "__main__":
    main()
