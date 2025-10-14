-- Select the swarm
SELECT
    swarm,
    cfg,
    serving_handle
FROM
    swarm
WHERE
    deployment_name = :deployment_name;
