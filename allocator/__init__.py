"""
allocator/
Decentralised task-allocation layer for the drone swarm.

Phase 1: interface only.
Phase 2: GreedyAuction, CBBA.
Phase 3: LearnedBidder (PPO-trained), OracleAllocator, BidPolicy.
"""

from allocator.base_allocator import BaseAllocator, AllocationResult, Bid, WorldSnapshot
from allocator.greedy_auction import GreedyAuction
from allocator.cbba import CBBA
from allocator.oracle import OracleAllocator
from allocator.bid_policy import BidPolicy, build_bid_obs, BID_OBS_DIM
from allocator.learned_bidder import LearnedBidder

__all__ = [
    "BaseAllocator", "AllocationResult", "Bid", "WorldSnapshot",
    "GreedyAuction",
    "CBBA",
    "OracleAllocator",
    "BidPolicy", "build_bid_obs", "BID_OBS_DIM",
    "LearnedBidder",
]
