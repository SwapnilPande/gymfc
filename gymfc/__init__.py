from gym.envs.registration import register

register(
    id='attitude-fc-v0',
    entry_point='gymfc.envs:AttitudeFlightControlEnv',
)