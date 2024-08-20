import numpy as np
import mlx.core as mx
from ..inference_engine import InferenceEngine
from .sharded_model import StatefulShardedModel
from .sharded_utils import load_shard
from ..shard import Shard
from typing import Optional
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from uuid import uuid4
import socket

class MLXDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self):
        self.shard = None
        self.collect_topology = GRPCPeerHandle(id=str(uuid4()),address=socket.getaddrinfo(host="localhost:8081"), port="8081")

    async def infer_prompt(self, shard: Shard, prompt: str, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
        await self.ensure_shard(shard)
        output_data: np.ndarray = np.array(self.stateful_sharded_model.step(mx.array(self.tokenizer.encode(prompt))))
        return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

    async def infer_tensor(self, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
        await self.ensure_shard(shard)
        output_data: np.ndarray = np.array(self.stateful_sharded_model.step(mx.array(input_data)))
        return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

    async def reset_shard(self, shard: Shard):
        await self.ensure_shard(shard)
        self.stateful_sharded_model.reset()

    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return

        model_shard, self.tokenizer = await load_shard(shard.model_id, shard)
        self.stateful_sharded_model = StatefulShardedModel(shard, model_shard)
        self.shard = shard

    async def get_shard():
        pass