from multigrid import wrappers
from gymnasium import ObservationWrapper


class MultiAgentImgObsWrapper(ObservationWrapper):
    """
    A wrapper for multi-agent environments to return only image observations for each agent.
    Each agent's observation is a key-value pair in a dictionary.
    """

    def __init__(self, env):
        """A wrapper that extracts only the image observation for each agent in a multi-agent environment.

        Args:
            env: The multi-agent environment to apply the wrapper
        """
        super().__init__(env)

        # Adjust this based on your environment's specific observation space structure
        self.observation_space = {agent_id: env.observation_space[agent_id].spaces["image"] 
                                  for agent_id in env.observation_space}

    def observation(self, obs):
        """Extract and return the image observation for each agent.

        Args:
            obs: The original observations from the multi-agent environment, structured as a dictionary.

        Returns:
            A dictionary with the same keys (agent identifiers) and their corresponding image observations.
        """
        return {agent_id: agent_obs["image"] for agent_id, agent_obs in obs.items()}