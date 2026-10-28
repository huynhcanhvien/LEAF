"""The forecast D of future class-token attention.

  future_attention    folded query-key forms of all later blocks (reference math)
  head_groups         groups of (block, head) rows by class-token attention pattern (group_heads, once per backbone)
  attention_forecast  AttentionForecast: grouped log-domain D truncated to rank r
  kernels/            Triton kernels of the station score (1 - alpha) z(log P) + alpha z(D) (station_score)
  cpu_forecast        the same computation on the CPU, for CPU timing
"""
