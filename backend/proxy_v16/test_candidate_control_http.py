"""CPU loopback tests for the combined Letta candidate cancellation contract.

The upstream model is an artificial fixture.  These tests verify ownership and
terminal accounting only; they are not GPU performance or agent-quality data.
"""
import asyncio
import json
import unittest

from test_cancellation_http import AbortHarness, until


class CandidateControlHTTP(unittest.IsolatedAsyncioTestCase):
    async def harness(self):
        # The test withholds native termination until cancel returns 202.
        # Its control wait must finish well before the fixture's 5s upstream
        # request timeout, rather than racing that timeout at the same 5s.
        h = await AbortHarness(delay_headers=True).start(candidate_terminal_wait=0.25)
        self.addAsyncCleanup(h.close)
        return h

    @staticmethod
    def headers(candidate_id, rid):
        return {
            "x-toolslack-request-id": rid,
            "x-toolslack-request-kind": "memory",
            "x-toolslack-session-id": "conv-cpu-fixture",
            "x-toolslack-candidate-id": candidate_id,
            "x-toolslack-native-request-id": rid,
            "x-toolslack-native-request-kind": "native-memory",
        }

    async def start_native(self, h, candidate_id="candidate-cpu", rid="memory-cpu"):
        async def request():
            async with h.client.post(
                h.url + "/v1/chat/completions",
                json=h.body(stream=True),
                headers=self.headers(candidate_id, rid),
            ) as response:
                return response.status, await response.text()

        task = asyncio.create_task(request())
        await asyncio.wait_for(h.chat_arrived.wait(), 2)
        return task

    async def cancel(self, h, candidate_id):
        return await h.client.post(
            h.url + f"/toolslack/v1/candidates/{candidate_id}/cancel",
            json={"reason": "tool_result_ready"},
        )

    async def test_abort_ack_is_not_candidate_terminal(self):
        h = await self.harness()
        task = await self.start_native(h)
        response = await self.cancel(h, "candidate-cpu")
        snapshot = await response.json()
        self.assertEqual(response.status, 202)
        self.assertTrue(snapshot["cancelled"])
        self.assertFalse(snapshot["terminal"])
        self.assertFalse(snapshot["budget_releasable"])
        self.assertEqual(snapshot["requests"][0]["rid"], "memory-cpu")
        self.assertFalse(snapshot["requests"][0]["terminal_confirmed"])
        self.assertGreaterEqual(snapshot["requests"][0]["abort_enqueued"], 1)
        self.assertEqual(h.release_calls, 0)

        h.allow_terminal.set()
        self.assertEqual((await task)[0], 200)
        await until(lambda: not h.refs)
        async with h.client.get(
            h.url + "/toolslack/v1/candidates/candidate-cpu"
        ) as status_response:
            final = await status_response.json()
        self.assertTrue(final["terminal"])
        self.assertTrue(final["budget_releasable"])
        self.assertTrue(final["requests"][0]["terminal_confirmed"])

    async def test_tombstone_rejects_future_native_id(self):
        h = await self.harness()
        response = await self.cancel(h, "candidate-before-submit")
        snapshot = await response.json()
        self.assertEqual(response.status, 200)
        self.assertTrue(snapshot["terminal"])
        self.assertEqual(snapshot["requests"], [])
        async with h.client.post(
            h.url + "/v1/chat/completions",
            json=h.body(),
            headers=self.headers("candidate-before-submit", "late-memory"),
        ) as late:
            self.assertEqual(late.status, 409)
        self.assertEqual(h.chat_calls, 0)

    async def test_incomplete_candidate_ownership_is_rejected_pre_submission(self):
        h = await self.harness()
        headers = self.headers("candidate-conflict", "rid-one")
        headers["x-toolslack-native-request-id"] = "rid-two"
        async with h.client.post(
            h.url + "/v1/chat/completions", json=h.body(), headers=headers
        ) as response:
            self.assertEqual(response.status, 400)
        self.assertEqual(h.chat_calls, 0)


if __name__ == "__main__":
    unittest.main()
