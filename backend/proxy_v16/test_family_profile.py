import copy
import json
import tempfile
import unittest
from pathlib import Path
from family_profile import load_profile
from model_proxy import normalize
from qwen_profile import ModelProfile as Qwen
from dsv4_profile import ModelProfile as DeepSeek


class FamilyContracts(unittest.TestCase):
    def test_native_content_and_generation_parameters_preserved(self):
        for profile in [Qwen(),DeepSeek()]:
            body=profile.body([{'role':'user','content':'完整原文 including tokens.'}],max_tokens=256)
            body.update(temperature=.2,top_p=.3,presence_penalty=.8,tools=[{'type':'function','function':{
                'name':'read','parameters':{'type':'object'}}}],tool_choice='none')
            old=copy.deepcopy(body)
            self.assertEqual(normalize(body,profile),old)
            self.assertEqual(body,old)

    def test_family_specific_template_key(self):
        for profile,bad in [(DeepSeek(),{'enable_thinking':False}),(Qwen(),{'thinking':False}),
                            (DeepSeek(),{'thinking':0}),(Qwen(),{'enable_thinking':True})]:
            body=profile.body([{'role':'user','content':'x'}]);body['chat_template_kwargs']=bad
            with self.assertRaises(ValueError):normalize(body,profile)

    def test_profile_roundtrip_and_actual_capacity90(self):
        with tempfile.TemporaryDirectory() as directory:
            for profile,trigger in [(Qwen(),234086),(DeepSeek(),941875)]:
                path=Path(directory)/'profile.json';path.write_text(json.dumps(profile.__dict__))
                actual=load_profile(path)
                self.assertEqual(actual,profile)
                self.assertEqual(actual.trigger_tokens,trigger)

    def test_native_output_cap_is_preserved_not_raised(self):
        for profile in [Qwen(),DeepSeek()]:
            body=profile.body([{'role':'user','content':'x'}]);body.pop('max_tokens')
            body['max_completion_tokens']=512
            result=normalize(body,profile)
            self.assertEqual(result['max_tokens'],512)
            self.assertNotIn('max_completion_tokens',result)


if __name__=='__main__':unittest.main()
