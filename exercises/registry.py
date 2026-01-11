from . import squat
from . import push_up
from . import pull_up
from . import deadlift
from . import chin_up

SKILLS = {
    "squat": squat,
    "push_up": push_up,
    "pull_up": pull_up,
    "deadlift": deadlift,
    "chin_up": chin_up,
}

def get_skill(name: str):
    return SKILLS[name]
