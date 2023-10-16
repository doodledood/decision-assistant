import random
from typing import Optional

from dotenv import load_dotenv
from fire import Fire
import autogen


def run_goal_assistant(llm_config: any):
    pass


def run_decision_assistant(goal: str, llm_temperature: float = 0.0, llm_seed: Optional[int] = None):
    if llm_seed is None:
        llm_seed = random.randint(0, 1000000)

    config_list = autogen.config_list_from_models(model_list=["gpt-4-0613"],
                                                  exclude="aoai")
    default_llm_config = dict(seed=llm_seed,
                              temperature=llm_temperature,
                              config_list=config_list,
                              model="gpt-4-0613",
                              request_timeout=120)

    print(config_list)


if __name__ == '__main__':
    load_dotenv()

    Fire(run_decision_assistant)
