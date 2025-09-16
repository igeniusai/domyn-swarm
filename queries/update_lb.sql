-- Update LB information
UPDATE swarm
SET
    lb_port = :lb_port,
    lb_node = :lb_node,
    endpoint = :endpoint
WHERE
    jobid = :jobid;