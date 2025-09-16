-- Delete a record from the swarm
DELETE FROM swarm
WHERE
    jobid = :jobid;
