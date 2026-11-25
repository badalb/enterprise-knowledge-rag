# Spark Performance

Synthetic documentation for a fictional enterprise.

## Shuffle and skew

Inspect Spark UI stage metrics for shuffle spill, task duration and skew. Enable adaptive query execution where supported. Broadcast small dimension tables only when they fit executor memory.

## Partitioning

Avoid excessive small tasks and oversized partitions. Investigate garbage collection and executor memory overhead before increasing heap size.
