"""
toughness.py - Score dataset problems with the RouteLLM BERT router.
No model inference is performed — only difficulty scoring.
"""


ROUTER    = "bert"

def get_router_client(weak_model: str = "", strong_model: str = ""):
    """Return a RouteLLM Controller for the given models."""
    from routellm.controller import Controller
    return Controller(
        routers=[ROUTER],
        strong_model=f"openai/{strong_model}",
        weak_model=f"openai/{weak_model}",
    )


def record_toughness(
    split_id: int,
    is_test: bool = False,
) -> None:
    """Write per-problem difficulty scores to the DB for tasks not yet scored.

    Args:
        split_id: DB split id to score.
        is_test:  Whether to score the test partition (default: train).
    """
    from daos import tasks_dao

    tasks = tasks_dao.get_unscored_for_split(split_id, is_test=is_test)
    if not tasks:
        print("All tasks already scored, skipping.")
        return

    router_client = get_router_client()
    router = router_client.routers[ROUTER]

    for i, task in enumerate(tasks):
        score = router.calculate_strong_win_rate(task.prompt)
        tasks_dao.set_toughness_score(task.id, float(score))
        print(f"[{i + 1}] {task.id} -> {score:.4f}")

    print(f"\nDone. Scored {len(tasks)} tasks.")
