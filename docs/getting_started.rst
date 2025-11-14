Getting started
===============

Installation
------------

.. code-block:: bash

   pip install domyn-swarm
   # or with uv
   uv tool install --from git+ssh://git@github.com/igeniusai/domyn-swarm.git --python 3.12 domyn-swarm

Minimal example
---------------

.. code-block:: yaml

   # config.yaml
   name: "smollm3-3b"
   model: "HuggingFaceTB/SmolLM3-3B"
   image: "/shared/images/vllm.sif"
   replicas: 1
   gpus_per_node: 1
   gpus_per_replica: 1
   port: 9001
   backend:
     type: slurm
     partition: "gpu"
     account: "..."
     qos: "..."

Launch a swarm:

.. code-block:: bash

   SWARM_NAME=$(domyn-swarm up -c config.yaml)
   echo "$SWARM_NAME"

Then submit a job:

.. code-block:: bash

   domyn-swarm job submit --swarm "$SWARM_NAME" ...

Finally shut it down:

.. code-block:: bash

   domyn-swarm down "$SWARM_NAME"