import pytest

from avatar.llm.mock import MockProvider
from avatar.llm.scripted import ScriptedProvider
from avatar.pipeline import Pipeline
from avatar.storage import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def episode(store):
    return store.add_episode(
        title="Weekly run",
        situation='A friend texted: "Want to join our Tuesday run? It\'s every week."',
        immediate_thought="Now I have to do this every week.",
        emotion_body="Irritation. My jaw went tight.",
        action_outcome="I left it unanswered for two days.",
        comparable_occasion="I run on my own whenever I feel like it and enjoy it.",
    )


@pytest.fixture
def scripted():
    return ScriptedProvider()


@pytest.fixture
def pipeline(store, scripted):
    return Pipeline(store, scripted)


@pytest.fixture
def mock_pipeline(store):
    return Pipeline(store, MockProvider())
