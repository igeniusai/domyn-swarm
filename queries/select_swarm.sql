-- Select the swarm
SELECT
    name,
    jobid,
    lb_jobid,
    lb_node,
    endpoint,
    delete_on_exit,
    model
FROM
    swarm
WHERE
    jobid = :jobid;
