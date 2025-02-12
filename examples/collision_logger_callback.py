from ray.rllib.agents.callbacks import DefaultCallbacks
from flow.examples.warning_logger import collision_logger

class CollisionLoggerCallbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs):
        try:
            # local worker의 환경 인스턴스를 가져와 collision 수 집계
            env = trainer.workers.local_worker().env
            coll_count = collision_logger(env)
        except Exception as e:
            coll_count = 0
            print(f"[ERROR] CollisionLoggerCallbacks: {e}")
        result.setdefault("custom_metrics", {})
        result["custom_metrics"]["collision_count"] = coll_count