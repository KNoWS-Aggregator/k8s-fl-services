"""Training services participating in federated sessions.

Replace or extend these base URLs for the target deployment. Protocol paths
such as /session/start and /train are appended by the coordinator.
"""

TRAINING_CLIENT_BASE_URLS = [
    "http://model-training-1:8080",
    "http://model-training-2:8080",
]
