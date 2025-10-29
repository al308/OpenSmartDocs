import json as json_module
from dataclasses import dataclass

import pytest

from onedrive_ollama_pipeline.config import OllamaSettings
from onedrive_ollama_pipeline.ollama_client import OllamaClient


class FakeCompletion:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class StubCompletions:
    def __init__(self, generator):
        self.generator = generator
        self.calls = []

    def create(self, **kwargs):  # noqa: D401 - mimic OpenAI API
        self.calls.append(kwargs)
        payload = self.generator(kwargs)
        return FakeCompletion(payload)


@dataclass
class FakeClient:
    completions: StubCompletions

    def __post_init__(self):
        self.chat = type('Chat', (), {'completions': self.completions})()


def test_request_metadata_returns_json():
    def generator(kwargs):
        metadata_json = json_module.dumps(
            {
                'title': 'Sample',
                'author': 'Tester',
                'document_type': 'invoice',
                'summary': 'Sample summary',
                'tags': ['invoice'],
                'language': 'de',
            }
        )
        return {'choices': [{'message': {'content': metadata_json}}]}

    completions = StubCompletions(generator)
    client = FakeClient(completions)
    settings = OllamaSettings(base_url='http://localhost:1234/v1', model='test-model')
    ollama = OllamaClient(settings, client=client)

    metadata = ollama.request_metadata(b'binary-image')

    assert metadata['title'] == 'Sample'
    call = completions.calls[0]
    assert call['model'] == 'test-model'
    assert call['timeout'] == 120
    encoded_image = call['messages'][1]['content'][1]['image_url']['url']
    assert encoded_image.startswith('data:image/png;base64,')


def test_request_metadata_invalid_json():
    def generator(_kwargs):
        return {'choices': [{'message': {'content': 'not json'}}]}

    completions = StubCompletions(generator)
    client = FakeClient(completions)
    settings = OllamaSettings(base_url='http://localhost:1234/v1', model='test-model')
    ollama = OllamaClient(settings, client=client)

    with pytest.raises(RuntimeError):
        ollama.request_metadata(b'binary-image')


def test_ensure_ready_success():
    def generator(_kwargs):
        return {'choices': [{'message': {'content': 'test'}}]}

    completions = StubCompletions(generator)
    client = FakeClient(completions)
    settings = OllamaSettings(base_url='http://localhost:1234/v1', model='test-model')
    ollama = OllamaClient(settings, client=client)

    ollama.ensure_ready()
    assert completions.calls[0]['timeout'] == 30


def test_ensure_ready_failure():
    def generator(_kwargs):
        return {'choices': [{'message': {'content': 'unexpected'}}]}

    completions = StubCompletions(generator)
    client = FakeClient(completions)
    settings = OllamaSettings(base_url='http://localhost:1234/v1', model='test-model')
    ollama = OllamaClient(settings, client=client)

    with pytest.raises(RuntimeError):
        ollama.ensure_ready()


def test_request_structure_plan_uses_json_response_format():
    def generator(_kwargs):
        return {'choices': [{'message': {'content': json_module.dumps({'summary': 'ok', 'operations': []})}}]}

    completions = StubCompletions(generator)
    client = FakeClient(completions)
    settings = OllamaSettings(base_url='http://localhost:11434/v1', model='llama3')
    ollama = OllamaClient(settings, client=client)

    result = ollama.request_structure_plan('prompt', model='llama3')

    assert json_module.loads(result)['summary'] == 'ok'
    call = completions.calls[0]
    assert call['response_format'] == {'type': 'json_object'}
    assert call['model'] == 'llama3'


def test_request_structure_plan_with_schema():
    def generator(_kwargs):
        return {'choices': [{'message': {'content': json_module.dumps({'summary': 'ok', 'operations': []})}}]}

    completions = StubCompletions(generator)
    client = FakeClient(completions)
    settings = OllamaSettings(base_url='http://localhost:11434/v1', model='llama3')
    ollama = OllamaClient(settings, client=client)

    schema = {
        'type': 'object',
        'properties': {
            'summary': {'type': 'string'},
            'operations': {'type': 'array'},
        },
        'required': ['summary', 'operations'],
    }
    result = ollama.request_structure_plan('prompt', model='llama3', json_schema=schema, schema_name='StructurePlanTest')

    assert json_module.loads(result)['summary'] == 'ok'
    call = completions.calls[0]
    assert call['response_format'] == {
        'type': 'json_schema',
        'json_schema': {'name': 'StructurePlanTest', 'schema': schema},
    }
    assert call['model'] == 'llama3'
