# Kafka Troubleshooting

Synthetic documentation for a fictional enterprise.

## Consumer lag

Consumer lag is the difference between the latest partition offset and the consumer group committed offset. Investigate processing latency, partition assignment, downstream dependencies and consumer errors.

## Rebalancing and broker health

Frequent rebalances can interrupt processing. Check max.poll.interval.ms, consumer restarts and group membership. Inspect under-replicated partitions and ISR changes. Never reset offsets without approval.
