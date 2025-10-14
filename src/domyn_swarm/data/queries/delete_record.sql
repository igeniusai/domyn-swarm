-- Delete a record from the swarm
DELETE FROM swarm
WHERE
    deployment_name = :deployment_name;
