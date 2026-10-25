"""Latency.

  salt           SALT, the station-aware latency model: features, non-negative fit (fit_salt), cost for the planner
  timing         latency of a model as one CUDA graph (measure_latency)
  cpu_timing     latency on the CPU (measure_latency_cpu)
  workloads      timing workloads for fitting SALT on a new backbone or GPU
  profile_model  the earlier profile latency model of LEAF-M0
"""
