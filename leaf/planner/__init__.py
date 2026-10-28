"""Schedule planning.

  probes         single-station probe grid (probe_jobs, probe_state); also the earlier path-DP Planner of LEAF-M0
  surrogate      probe curves and the separable distortion tables (ProbeCurve, JointTables, lookup)
  lagrange       the exact Lagrangian DP over (block, token count) states (lagrange_dp)
  plan           plan_latency: SALT price, budget bisection, eta rescaling; operator_config, config_details
  gmac           GMAC accounting (config_cost) and the GMAC-budget planner (GmacPlanner)
  budgets        latency budgets from measured anchors (the paper's rule)
  configs        planner state <-> config (ForecastConfig, config_from_state, DiffRate schedules)
  schedule_caps  token schedules: per-block (prune, merge) caps <-> stations, GMAC of a schedule
"""
