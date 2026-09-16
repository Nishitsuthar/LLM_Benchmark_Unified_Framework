"""Offline final45 integration tests; provider construction and networking prohibited."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run
from unified_pipeline.base import ModelResponse
from unified_pipeline.evidence.visual_pdf_evidence import VisualPdfEvidenceBuilder
from scripts.validate_imdb_final45 import validate

class Final45Tests(unittest.TestCase):
    def test_frozen_artifacts_and_independent_answers(self):
        self.assertEqual(validate()['questions_verified'],45)

    def test_builder_rejects_invalid_layout_and_cast_for_metadata_only(self):
        import yaml
        c=yaml.safe_load(Path('datasets/imdb_20_final45.yaml').read_text())
        for changes in [{'page_layout':'invalid','_source_pdf':'metadata'},{'_source_pdf':'cast'},{'pages_per_image':3,'_source_pdf':'metadata'}]:
            with self.assertRaises(ValueError):VisualPdfEvidenceBuilder().build('',dict(c,**changes))

    def test_cli_guard_covers_final45(self):
        with patch('sys.argv',['run.py','--dataset','imdb_20_final45','--model','gemini_flash']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):run.parse_args()

    def test_both_models_receive_identical_five_image_payloads(self):
        seen=[]
        payloads=[]
        expected=run.load_questions('evidence/imdb_20_final45/questions.json')
        template=Path('prompts/imdb_20_final45_llm_only_zero_shot.txt').read_text()
        allowed={template.format(question=q['text'],answer_format=q['answer_format']) for q in expected}
        class FakeRouter:
            def __init__(self,model):
                self.model_name='offline-stub';self.last_raw_response=None;self.api_successes=0;self.api_failures=0
                self.client=type('FakeClient',(),{'with_options':lambda self,**kw:self})()
            def call(self,prompt_text,images=None):
                assert prompt_text in allowed
                assert len(images)==5 and all(v.startswith('data:image/jpeg;base64,') for v in images)
                payloads.append(tuple(images))
                seen.append(prompt_text);self.api_successes+=1
                self.last_raw_response={'choices':[{'message':{'content':'OFFLINE TEST'}}]}
                return ModelResponse('OFFLINE TEST','','stop',0,0,0)
        with tempfile.TemporaryDirectory() as directory,patch.object(run,'RESULTS_DIR',Path(directory)),patch.object(run,'ModelRouter',FakeRouter),patch('socket.socket',side_effect=AssertionError('Network forbidden')),patch('dotenv.load_dotenv',side_effect=AssertionError('.env must not be read')):
            for model,ids in [('gemini_flash',[]),('mistral_small',[])]:
                argv=['run.py','--dataset','imdb_20_final45','--model',model,'--confirm-full-run']
                if ids:argv+=['--task_ids',*ids]
                with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):run.main()
            self.assertEqual(len(seen),90)
            self.assertEqual(len(set(payloads)),1)
            self.assertEqual(set(seen[45:]),{template.format(question=q['text'],answer_format=q['answer_format']) for q in expected})
            summaries=list(Path(directory).glob('*/imdb_20_final45/*/*summary.json'))
            self.assertEqual(len(summaries),2)
            self.assertEqual(sorted(json.loads(p.read_text())['attempted_questions'] for p in summaries),[45,45])
            self.assertFalse((Path(directory)/'gemini_flash/imdb_20').exists())

if __name__=='__main__':unittest.main()
