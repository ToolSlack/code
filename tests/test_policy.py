import unittest
from dataclasses import replace

from toolslack.policy import (LevelSelector, ProfileKey, ProfileStore, ReadyTask,
                             ResourceCaps, ToolSignature, TwoQueueScheduler,
                             WindowPredictor)
from toolslack.types import Plan, Scope


KEY = ProfileKey("model-A", "native", "out=256", "workers=1", "cold", "workload-A")
SIGNATURE = ToolSignature("model-A", "pytest", "suite=integration", "env-A", "workload-A")


class PolicyInvariantTests(unittest.TestCase):
    def profiles(self):
        profiles = ProfileStore(bucket_size=1)
        for tokens, seconds in [(100, 1), (200, 2), (300, 3)]:
            for _ in range(3):
                profiles.observe_cost("memory", tokens, seconds, KEY)
                profiles.observe_cost("kv", tokens, seconds / 2, KEY)
        return profiles

    def prediction(self, duration=10):
        predictor = WindowPredictor()
        for _ in range(3):
            predictor.observe(SIGNATURE, duration)
        return predictor.predict(SIGNATURE, 0)

    def plan(self, gain=1, deadline=10, initial=10, memory=1, kv=0, level=1):
        scope = Scope(1, 100, 100, .5, 1)
        return Plan(level, scope, 100 if kv else 0, memory, kv, gain,
                    deadline, initial, 2)

    def test_uncertain_or_foreign_tool_history_cannot_start_work(self):
        predictor = WindowPredictor()
        predictor.observe(SIGNATURE, 10)
        predictor.observe(SIGNATURE, 10)
        selector = LevelSelector(self.profiles(), KEY)
        scopes = [Scope(1, 100, 100, 1, 2)]
        self.assertIsNone(selector.select(scopes, predictor.predict(SIGNATURE, 0), 0))
        calibrated = predictor.predict(SIGNATURE, 0, explicit_calibration_s=3)
        self.assertIsNotNone(selector.select(scopes, calibrated, 0))
        foreign = replace(SIGNATURE, environment_key="env-B")
        self.assertFalse(predictor.predict(foreign, 0).usable)
        foreign = replace(SIGNATURE, model="model-B")
        for _ in range(3):
            predictor.observe(foreign, 10)
        self.assertIsNone(selector.select(scopes, predictor.predict(foreign, 0), 0))
        snapshot = predictor.predict(SIGNATURE, 0, explicit_calibration_s=3)
        predictor.observe(SIGNATURE, .1)
        self.assertEqual(snapshot.duration_s, 3)
        self.assertIsNone(selector.select(scopes, snapshot, -1))

    def test_cost_fit_keeps_launch_overhead_and_never_decreases(self):
        profiles = ProfileStore(bucket_size=1)
        for seconds in [3, 3.5, 4]:
            profiles.observe_cost("memory", 100, seconds, KEY)
        for seconds in [1, 2, 2]:
            profiles.observe_cost("memory", 200, seconds, KEY)
        self.assertEqual(profiles.predict_cost("memory", 1, KEY).service_s, 4)
        costs = [profiles.predict_cost("memory", n, KEY).service_s for n in [1, 100, 150, 200, 300]]
        self.assertEqual(costs, sorted(costs))
        self.assertIsNone(profiles.predict_cost("memory", 100, replace(KEY, cache_key="warm")))
        self.assertIsNone(profiles.predict_cost("kv", 100, KEY))
        with self.assertRaises(ValueError):
            profiles.observe_cost("memory", 100, -1, KEY)

    def test_select_best_gain_even_when_scope_benefit_is_nonmonotonic(self):
        selector = LevelSelector(self.profiles(), KEY)
        scopes = [Scope(1, 100, 100, 2, 2), Scope(2, 200, 200, 8, 8),
                  Scope(3, 300, 300, 3, 3)]
        plan = selector.select(scopes, self.prediction(4), 0, .25)
        self.assertEqual(plan.scope.stop, 2)
        self.assertLessEqual(plan.memory_s, 4 - .25)
        # Smaller scopes may have greater benefit than the full native scope.
        full = LevelSelector(self.profiles(), KEY, no_selector=True)
        self.assertEqual(full.select(scopes, self.prediction(4), 0).scope.stop, 3)
        self.assertIsNone(full.select(scopes, self.prediction(2.5), 0))

    def test_joint_l2_benefit_can_outweigh_a_bigger_l1_scope(self):
        selector = LevelSelector(self.profiles(), KEY)
        scopes = [Scope(1, 100, 100, 1, 8), Scope(2, 200, 200, 5, 5)]
        plan = selector.select(scopes, self.prediction(2), 0, .1, kv_lengths=[100, 200])
        self.assertEqual((plan.level, plan.scope.stop, plan.kv_tokens), (2, 1, 100))
        l1 = LevelSelector(self.profiles(), KEY, no_kv=True).select(
            scopes, self.prediction(2), 0, .1, kv_lengths=[100])
        self.assertEqual((l1.level, l1.scope.stop), (1, 1))

    def test_kv_rebudget_is_only_remaining_work_and_uses_actual_prefix(self):
        selector = LevelSelector(self.profiles(), KEY)
        scope = Scope(1, 100, 100, 1, 4)
        plan = selector.select([scope], self.prediction(4), 0, .2, kv_lengths=[100])
        proposal = selector.rebudget_kv(plan, 300, 3, .2, kv_lengths=[100, 200, 300], queue_kv_s=0)
        self.assertEqual(proposal.kv_tokens, 100)
        self.assertEqual(proposal.memory_s, 0)
        self.assertEqual(proposal.initial_budget, plan.initial_budget)
        self.assertEqual(proposal.deadline, plan.deadline)
        self.assertEqual(proposal.scope.new_prefix_tokens, 300)
        task = ReadyTask("kv-ready", "kv", proposal, 3)
        self.assertTrue(TwoQueueScheduler().feasible(task, 3, .2))
        self.assertIsNone(selector.rebudget_kv(plan, 300, 3.4, .2, kv_lengths=[100]))

    def test_elapsed_queue_is_not_charged_at_dispatch(self):
        # Memory waited 2 seconds, which is already removed from the deadline.
        task = ReadyTask("memory", "memory", self.plan(deadline=5, initial=5, memory=1), 0)
        self.assertTrue(TwoQueueScheduler().feasible(task, 2, .5))
        # A downstream KV queue is future work and must still be counted.
        joint = replace(task, plan=self.plan(deadline=5, initial=5, memory=1, kv=1, level=2),
                        downstream_queue_s=1)
        self.assertFalse(TwoQueueScheduler().feasible(joint, 2, .5))
        selector = LevelSelector(self.profiles(), KEY)
        self.assertIsNone(selector.select([Scope(1, 100, 100, 1, 2)],
                                         self.prediction(2), 0, queue_memory_s=2))

    def test_urgent_and_aged_tasks_precede_high_benefit_without_bypassing_deadline(self):
        scheduler = TwoQueueScheduler()
        high = ReadyTask("high", "memory", self.plan(gain=2, deadline=12), 2.5)
        urgent = ReadyTask("urgent", "memory", self.plan(gain=.1, deadline=4.5), 2.5)
        aged = ReadyTask("aged", "memory", self.plan(gain=.01, deadline=12), 0)
        self.assertEqual(scheduler.choose([high, urgent], 3).task_id, "urgent")
        self.assertEqual(scheduler.choose([high, aged], 3).task_id, "aged")
        self.assertIsNone(scheduler.choose([urgent], 4.5))
        consumed = replace(high, consumed=True)
        invalid = replace(high, valid=False)
        self.assertEqual(scheduler.ordered([consumed, invalid], 3), [])

    def test_foreground_block_and_fifo_ablation_keep_feasibility_guards(self):
        high = ReadyTask("high", "memory", self.plan(gain=2), .5)
        low = ReadyTask("low", "memory", self.plan(gain=.1), 0)
        self.assertEqual(TwoQueueScheduler(fifo=True).choose([high, low], 1).task_id, "low")
        chosen = TwoQueueScheduler().choose([high, low], 1,
                                            can_dispatch=lambda task: task.task_id != "high")
        self.assertEqual(chosen.task_id, "low")
        expired = replace(low, plan=replace(low.plan, deadline=1))
        self.assertIsNone(TwoQueueScheduler(fifo=True).choose([expired], 1))
        self.assertIsNone(TwoQueueScheduler(fifo=True).choose([replace(low, admitted_at=2)], 1))
        with self.assertRaises(ValueError):
            TwoQueueScheduler().feasible(low, 1, -.1)
        selector = LevelSelector(self.profiles(), KEY)
        self.assertIsNone(selector.select([Scope(1, 100, 100, 1, 2)], self.prediction(), 0,
                                         caps=ResourceCaps(foreground_available=False)))


if __name__ == "__main__":
    unittest.main()
