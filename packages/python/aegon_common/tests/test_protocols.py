from collections.abc import Sequence
from datetime import UTC, datetime

from aegon_common import Embedder, EmbeddingVector, IdGenerator, TextClock


class FakeEmbedder:
    model_id = "fake"
    dim = 2
    version = "1"

    async def embed_texts(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        return [
            EmbeddingVector((float(len(text)), 0.0), self.model_id, self.version) for text in texts
        ]

    async def embed_images(self, images: Sequence[bytes]) -> Sequence[EmbeddingVector]:
        return [
            EmbeddingVector((float(len(image)), 0.0), self.model_id, self.version)
            for image in images
        ]


class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)


class FixedIdGenerator:
    def new(self) -> str:
        return "fixed-id"


def test_fakes_satisfy_dependency_protocols() -> None:
    embedder: Embedder = FakeEmbedder()
    clock: TextClock = FixedClock()
    id_generator: IdGenerator = FixedIdGenerator()

    assert embedder.dim == 2
    assert clock.now().tzinfo is UTC
    assert id_generator.new() == "fixed-id"
