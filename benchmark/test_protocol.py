import json
from pathlib import Path
import tempfile
import unittest
import asyncio
import importlib.util
from benchmark.langgraph_hotpot import (EventLog, HTTPTransport, cpu_evidence,
    CalibratedBenefitSelector, apply_benefit_reference,
    count_raw_checkpoint_wire, frozen_policy, read_tasks)
from toolslack.langgraph_adapter import LangMemSettings
from toolslack.policy import ProfileKey, ProfileStore, ToolSignature, WindowPrediction
from toolslack.types import Scope
from dataclasses import asdict


class ProtocolTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "official LangGraph runtime required")
    def test_graph_schema_passes_native_cache_headers_to_next_foreground_node(self):
        from langgraph.graph import StateGraph,START,END
        from benchmark.langgraph_hotpot import AgentState
        headers={"x-toolslack-kv-handle":"verified-native-handle","x-toolslack-kv-one-shot":"true"}
        seen=[]
        async def tools(state):
            return dict(request_metadata={"headers":headers})
        async def answer(state):
            seen.append(state["request_metadata"]["headers"])
            return dict(prediction={"answer":"verified"})
        graph=StateGraph(AgentState)
        graph.add_node("tools",tools);graph.add_node("answer",answer)
        graph.add_edge(START,"tools");graph.add_edge("tools","answer");graph.add_edge("answer",END)
        asyncio.run(graph.compile().ainvoke(dict(task_id="schema-test")))
        self.assertEqual(seen,[headers])

    def test_owned_engine_flush_accepts_native_text_and_rejects_unknown_or_declined_receipts(self):
        from benchmark.run_series import flush_receipt
        class Response:
            def __init__(self,text,value=None):
                self.text,self.value=text,value
            def raise_for_status(self):
                pass
            def json(self):
                if self.value is None:
                    raise ValueError("native text")
                return self.value
        receipt=flush_receipt(Response("Cache flushed.\nPlease check backend logs for more details.\n"))
        self.assertIn("native_text_receipt",receipt)
        for response in (Response("busy"),Response("",False),Response("",{"ok":False})):
            with self.assertRaises(RuntimeError):
                flush_receipt(response)

    def test_serial_reference_is_full_observed_native_plus_consumer_and_keeps_negative_gains(self):
        rows=[dict(task_id="cal",scope_fraction=1.,scope_start=1,scope_stop=5,
                   memory_s=4.,original_ttft_s=1.,l1_ttft_s=2.,l2_ttft_s=1.5),
              dict(task_id="cal",scope_fraction=.5,scope_start=1,scope_stop=3,
                   memory_s=1.,original_ttft_s=1.,l1_ttft_s=7.,l2_ttft_s=6.5),
              dict(task_id="no-full",scope_fraction=.5,scope_start=1,scope_stop=3,
                   memory_s=1.,original_ttft_s=1.,l1_ttft_s=2.,l2_ttft_s=2.)]
        apply_benefit_reference(rows,"serial_native_memory_to_first_token")
        self.assertEqual(rows[0]["baseline_ttft_s"],6.)
        self.assertEqual(rows[0]["gain_l1_s"],4.)
        self.assertEqual(rows[0]["raw_original_gain_l1_s"],-1.)
        self.assertEqual(rows[1]["gain_l1_s"],-1.)
        self.assertFalse(rows[2]["benefit_valid"])

    def test_selector_uses_measured_serial_baseline_and_missing_reference_admits_nothing(self):
        key=ProfileKey("m","native","settings","load","cache","work")
        profile=ProfileStore();profile.observe_cost("memory",100,1.,key)
        selector=CalibratedBenefitSelector(profile,key,no_kv=True)
        scope=Scope(2,100,20,4.,4.)
        prediction=WindowPrediction(ToolSignature("m","tool","args","env","work"),0.,10.,3)
        self.assertIsNone(selector.select([scope],prediction,now=0.))
        selector.bind_baseline(scope,6.)
        plan=selector.select([scope],prediction,now=0.)
        self.assertEqual(plan.baseline_ttft_s,6.)
        self.assertEqual(plan.gain_s,4.)

    def test_native_system_summary_body_and_token_count_use_identical_raw_wire(self):
        class Tokenizer:
            chat_template="native"
            def apply_chat_template(self,messages,**options):
                self.received=messages
                return list(range(len(messages)*10))
        messages=[dict(role="system",content="protected instructions"),
                  dict(role="system",content="official summary"),
                  dict(role="user",content="actual question")]
        transport=HTTPTransport.__new__(HTTPTransport)
        transport.config=dict(model="Qwen3-8B");transport.tokenizer=Tokenizer()
        body=transport.body(messages,kind="answer")
        self.assertEqual(body["messages"],messages)
        self.assertEqual(transport.count(body["messages"]),30)
        self.assertEqual(transport.tokenizer.received,body["messages"])
        self.assertEqual(len(messages),3)

    def test_cpu_tools_preserve_native_titles_and_sentence_ids_without_gold(self):
        context=[["Real article",["The author was born in 1900.","A second sentence."]],
                 ["Distractor",["Unrelated detail."]]]
        result=cpu_evidence(context,"author birth")
        self.assertIn(["Real article",0],result["supporting_facts"])
        self.assertIn("1900",result["findings"])
        self.assertTrue(all(isinstance(i,int) for _,i in result["supporting_facts"]))

    def test_controller_monotonic_event_passes_through_without_duplicate_keyword(self):
        with tempfile.TemporaryDirectory() as directory:
            log=EventLog(Path(directory)/"events.jsonl")
            log(dict(event="controller_event",monotonic_s=123.4))
            log.close()
            self.assertEqual(json.loads(log.path.read_text())["monotonic_s"],123.4)

    def test_task_input_excludes_gold_and_requires_disjoint_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"agent_inputs";path.mkdir()
            (path/"distractor.jsonl").write_text(json.dumps(dict(_id="eval",question="q",context=[],answer="SECRET"))+"\n")
            config=dict(dataset_root=directory,calibration_task_ids=["cal"],evaluation_task_ids=["eval"])
            rows=read_tasks(config,"evaluation")
            self.assertEqual(set(rows[0]),{"_id","question","context"})
            config["calibration_task_ids"]=["eval"]
            with self.assertRaises(ValueError):
                read_tasks(config,"evaluation")

    def test_frozen_calibration_rejects_leakage_and_changed_native_algorithm_settings(self):
        config=dict(model="Qwen3-8B",service_profile_sha256="a"*64,evaluation_task_ids=["eval"])
        calibration=dict(model="Qwen3-8B",service_profile_sha256="a"*64,
            native_memory_settings=asdict(LangMemSettings()),task_ids=["cal"],tool_mode="cpu",
            scope_observations=[],tool_observations=[])
        selector,predictor=frozen_policy(config,calibration,"full")
        self.assertIsNotNone(selector)
        calibration["task_ids"]=["eval"]
        with self.assertRaises(ValueError):
            frozen_policy(config,calibration,"full")
        calibration["task_ids"]=["cal"];config["memory_trigger_tokens"]=1
        with self.assertRaises(ValueError):
            frozen_policy(config,calibration,"full")


if __name__=="__main__":
    unittest.main()
