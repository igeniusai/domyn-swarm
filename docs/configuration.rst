Configuration
=============

Domyn-swarm is configured via a YAML file that maps 1:1 to the
``DomynLLMSwarmConfig`` Pydantic model.

High-level schema
-----------------

.. automodule:: domyn_swarm.config
   :members: DomynLLMSwarmConfig, SlurmConfig, SlurmEndpointConfig, LeptonConfig
   :undoc-members:

Examples
--------

Slurm + Singularity
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/configs/smollm3.yaml
   :language: yaml
   :caption: Minimal config for SmolLM3 on Slurm with Singularity.