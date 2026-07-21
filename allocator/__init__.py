"""
allocator/
Decentralised task-allocation layer for the drone swarm.

Phase 1: interface only.
Phase 2: GreedyAuction, CBBA.
Phase 3: LearnedBidder (PPO-trained).
"""

from allocator.base_allocator import BaseAllocator, AllocationResult, Bid, WorldSnapshot
from allocator.greedy_auction import GreedyAuction
from allocator.cbba import CBBA

__all__ = [
    "BaseAllocator", "AllocationResult", "Bid", "WorldSnapshot",
    "GreedyAuction",
    "CBBA",
]
