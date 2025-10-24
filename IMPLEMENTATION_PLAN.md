# RayWhisper2 Implementation Plan

## Project Overview
A cross-platform (Windows/macOS) voice-to-text tool with RAG-enhanced transcription using state-of-the-art speech recognition, vector database embeddings, and keyboard-driven interaction.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Implementation Phases](#implementation-phases)
- [QA Strategy](#qa-strategy)
- [Configuration Specifications](#configuration-specifications)
- [Risk Mitigation](#risk-mitigation)
- [Progress Tracking](#progress-tracking)

---

## Architecture Overview

### Clean Architecture Layers

```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│    (CLI, System Tray, Keyboard UI)      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│        Application Layer                │
│  (Use Cases, Application Services)      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│          Domain Layer                   │
│   (Entities, Value Objects, Ports)      │
└─────────────────────────────────────────┘
                  ↑
┌─────────────────────────────────────────┐
│      Infrastructure Layer               │
│ (Whisper, VectorDB, Audio, Keyboard)    │
└─────────────────────────────────────────┘
```

### Vertical Slices (Features)

1. **Audio Recording Feature**: Capture audio from microphone
2. **Transcription Feature**: Convert audio to text using Whisper
3. **Embedding Management Feature**: Parse and embed documents
4. **RAG Retrieval Feature**: Retrieve and rerank context
5. **Keyboard Control Feature**: Global hotkeys for start/stop
6. **Output Simulation Feature**: Type text via keystrokes

---

## Technology Stack

### Core Technologies

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Speech-to-Text | `faster-whisper` | State-of-the-art, optimized Whisper implementation with CTranslate2 |
| Vector Database | `chromadb` | Lightweight, embeddable, excellent Python support |
| Embeddings | `sentence-transformers` | High-quality embeddings, local execution |
| Reranking | `sentence-transformers` (cross-encoder) | Accurate reranking with minimal latency |
| Audio Recording | `sounddevice` | Cross-platform, low-latency audio capture |
| Keyboard Control | `pynput` | Cross-platform keyboard hooks and simulation |
| Configuration | `pydantic` + `pydantic-settings` | Type-safe configuration with validation |
| Testing | `pytest` + `pytest-asyncio` | Comprehensive testing framework |
| Logging | `loguru` | Simple, powerful logging |

### Dependencies

```toml
# Core dependencies
faster-whisper = "^1.0.0"
chromadb = "^0.4.0"
sentence-transformers = "^2.2.0"
sounddevice = "^0.4.6"
numpy = "^1.24.0"
pynput = "^1.7.6"
pydantic = "^2.0.0"
pydantic-settings = "^2.0.0"
python-dotenv = "^1.0.0"
loguru = "^0.7.0"

# Development dependencies
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"
mypy = "^1.5.0"
ruff = "^0.1.0"
```

---

## Project Structure

```
raywhisper2/
├── src/
│   └── raywhisper/
│       ├── domain/                    # Core business logic (no external dependencies)
│       │   ├── entities/
│       │   │   ├── __init__.py
│       │   │   ├── transcription.py   # Transcription entity
│       │   │   ├── document.py        # Document entity
│       │   │   └── embedding.py       # Embedding entity
│       │   ├── value_objects/
│       │   │   ├── __init__.py
│       │   │   ├── audio_data.py      # Audio data value object
│       │   │   └── context.py         # Retrieved context value object
│       │   └── interfaces/            # Port interfaces
│       │       ├── __init__.py
│       │       ├── audio_recorder.py
│       │       ├── transcriber.py
│       │       ├── vector_store.py
│       │       ├── keyboard_controller.py
│       │       └── output_simulator.py
│       ├── application/               # Use cases and application services
│       │   ├── use_cases/
│       │   │   ├── __init__.py
│       │   │   ├── transcribe_audio.py
│       │   │   ├── populate_embeddings.py
│       │   │   ├── retrieve_context.py
│       │   │   └── process_voice_input.py
│       │   └── services/
│       │       ├── __init__.py
│       │       └── rag_service.py
│       ├── infrastructure/            # External implementations (adapters)
│       │   ├── audio/
│       │   │   ├── __init__.py
│       │   │   └── sounddevice_recorder.py
│       │   ├── transcription/
│       │   │   ├── __init__.py
│       │   │   └── whisper_transcriber.py
│       │   ├── embeddings/
│       │   │   ├── __init__.py
│       │   │   ├── document_parser.py
│       │   │   ├── embedding_generator.py
│       │   │   └── reranker.py
│       │   ├── vector_db/
│       │   │   ├── __init__.py
│       │   │   └── chroma_store.py
│       │   ├── keyboard/
│       │   │   ├── __init__.py
│       │   │   ├── base_controller.py
│       │   │   ├── windows_controller.py
│       │   │   └── macos_controller.py
│       │   └── output/
│       │       ├── __init__.py
│       │       └── keystroke_simulator.py
│       ├── presentation/              # UI/CLI layer
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   └── app.py                 # Main application orchestrator
│       └── config/
│           ├── __init__.py
│           └── settings.py            # Pydantic settings
├── tests/
│   ├── unit/
│   │   ├── domain/
│   │   ├── application/
│   │   └── infrastructure/
│   ├── integration/
│   └── e2e/
├── scripts/
│   └── populate_embeddings.py         # CLI tool for populating vector DB
├── config/
│   ├── config.yaml                    # Default configuration
│   └── config.example.yaml
├── .env.example
├── pyproject.toml
├── README.md
├── IMPLEMENTATION_PLAN.md
└── UserRequirements.md
```

---

## Implementation Phases

### Phase 1: Project Setup & Core Infrastructure
**Status**: ⬜ Not Started

#### Tasks
- [ ] 1.1 Initialize Python project with pyproject.toml
- [ ] 1.2 Set up project structure (directories)
- [ ] 1.3 Configure development tools (ruff, mypy, pytest)
- [ ] 1.4 Create configuration management system
- [ ] 1.5 Set up logging infrastructure
- [ ] 1.6 Create base domain interfaces

#### Pseudo-code: Configuration System

```python
# src/raywhisper/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal

class AudioSettings(BaseSettings):
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_duration: float = Field(default=0.1, description="Audio chunk duration in seconds")

class WhisperSettings(BaseSettings):
    model_size: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "base"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    compute_type: Literal["int8", "float16", "float32"] = "float16"
    language: str | None = None  # Auto-detect if None

class VectorDBSettings(BaseSettings):
    collection_name: str = "raywhisper_docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    persist_directory: str = "./data/chroma"
    top_k: int = 5

class RerankerSettings(BaseSettings):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 3

class KeyboardSettings(BaseSettings):
    start_stop_hotkey: str = "ctrl+shift+space"
    cancel_hotkey: str = "ctrl+shift+esc"

class OutputSettings(BaseSettings):
    typing_speed: float = 0.01  # Delay between keystrokes in seconds
    use_clipboard_fallback: bool = False

class Settings(BaseSettings):
    audio: AudioSettings = AudioSettings()
    whisper: WhisperSettings = WhisperSettings()
    vector_db: VectorDBSettings = VectorDBSettings()
    reranker: RerankerSettings = RerankerSettings()
    keyboard: KeyboardSettings = KeyboardSettings()
    output: OutputSettings = OutputSettings()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
```

#### QA Checklist
- [ ] Project structure follows Clean Architecture principles
- [ ] All dependencies install correctly on Windows and macOS
- [ ] Configuration loads from file and environment variables
- [ ] Logging outputs to console and file
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)

---

### Phase 2: Audio Recording Module
**Status**: ⬜ Not Started

#### Tasks
- [ ] 2.1 Define AudioRecorder interface (domain/interfaces)
- [ ] 2.2 Implement SoundDeviceRecorder (infrastructure)
- [ ] 2.3 Create AudioData value object
- [ ] 2.4 Implement audio buffer management
- [ ] 2.5 Add start/stop recording functionality
- [ ] 2.6 Write unit tests
- [ ] 2.7 Write integration tests

#### Pseudo-code: Audio Recording

```python
# src/raywhisper/domain/interfaces/audio_recorder.py
from abc import ABC, abstractmethod
from typing import Callable
from ..value_objects.audio_data import AudioData

class IAudioRecorder(ABC):
    @abstractmethod
    def start_recording(self, callback: Callable[[AudioData], None]) -> None:
        """Start recording audio and call callback with audio chunks."""
        pass

    @abstractmethod
    def stop_recording(self) -> AudioData:
        """Stop recording and return complete audio data."""
        pass

    @abstractmethod
    def is_recording(self) -> bool:
        """Check if currently recording."""
        pass

# src/raywhisper/domain/value_objects/audio_data.py
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    channels: int

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.samples) / self.sample_rate

    def to_wav_bytes(self) -> bytes:
        """Convert to WAV format bytes."""
        # Implementation here
        pass

# src/raywhisper/infrastructure/audio/sounddevice_recorder.py
import sounddevice as sd
import numpy as np
from queue import Queue
from threading import Event
from ...domain.interfaces.audio_recorder import IAudioRecorder
from ...domain.value_objects.audio_data import AudioData

class SoundDeviceRecorder(IAudioRecorder):
    def __init__(self, sample_rate: int, channels: int):
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_queue: Queue = Queue()
        self._recording_event = Event()
        self._stream = None

    def start_recording(self, callback=None) -> None:
        self._recording_event.set()
        self._audio_queue = Queue()

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            if self._recording_event.is_set():
                self._audio_queue.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            callback=audio_callback
        )
        self._stream.start()

    def stop_recording(self) -> AudioData:
        self._recording_event.clear()
        if self._stream:
            self._stream.stop()
            self._stream.close()

        # Collect all audio chunks
        chunks = []
        while not self._audio_queue.empty():
            chunks.append(self._audio_queue.get())

        if chunks:
            samples = np.concatenate(chunks, axis=0)
            return AudioData(
                samples=samples.flatten(),
                sample_rate=self._sample_rate,
                channels=self._channels
            )
        return AudioData(
            samples=np.array([]),
            sample_rate=self._sample_rate,
            channels=self._channels
        )

    def is_recording(self) -> bool:
        return self._recording_event.is_set()
```

#### QA Checklist
- [ ] Audio recording works on Windows
- [ ] Audio recording works on macOS
- [ ] Audio quality is sufficient (16kHz, mono)
- [ ] No audio dropouts or buffer overruns
- [ ] Memory usage is reasonable
- [ ] Unit tests cover all methods
- [ ] Integration tests verify actual recording

---

### Phase 3: Transcription Module
**Status**: ⬜ Not Started

#### Tasks
- [ ] 3.1 Define ITranscriber interface
- [ ] 3.2 Create Transcription entity
- [ ] 3.3 Implement WhisperTranscriber
- [ ] 3.4 Add model loading and caching
- [ ] 3.5 Implement audio preprocessing
- [ ] 3.6 Create TranscribeAudio use case
- [ ] 3.7 Write unit tests
- [ ] 3.8 Write integration tests
- [ ] 3.9 Performance benchmarking

#### Pseudo-code: Transcription

```python
# src/raywhisper/domain/interfaces/transcriber.py
from abc import ABC, abstractmethod
from ..value_objects.audio_data import AudioData
from ..entities.transcription import Transcription

class ITranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio: AudioData, context: str | None = None) -> Transcription:
        """Transcribe audio to text, optionally using context for guidance."""
        pass

# src/raywhisper/domain/entities/transcription.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Transcription:
    text: str
    language: str
    confidence: float
    timestamp: datetime
    context_used: str | None = None

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

# src/raywhisper/infrastructure/transcription/whisper_transcriber.py
from faster_whisper import WhisperModel
from ...domain.interfaces.transcriber import ITranscriber
from ...domain.value_objects.audio_data import AudioData
from ...domain.entities.transcription import Transcription
from datetime import datetime
import numpy as np

class WhisperTranscriber(ITranscriber):
    def __init__(self, model_size: str, device: str, compute_type: str):
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe(self, audio: AudioData, context: str | None = None) -> Transcription:
        # Prepare audio (faster-whisper expects float32 normalized to [-1, 1])
        audio_float = audio.samples.astype(np.float32)
        if audio_float.max() > 1.0:
            audio_float = audio_float / 32768.0  # Normalize int16 to float32

        # Transcribe with optional context
        segments, info = self._model.transcribe(
            audio_float,
            initial_prompt=context,
            beam_size=5,
            vad_filter=True  # Voice activity detection
        )

        # Combine all segments
        text = " ".join([segment.text for segment in segments])

        # Calculate average confidence
        segment_list = list(segments)
        avg_confidence = (
            sum(s.avg_logprob for s in segment_list) / len(segment_list)
            if segment_list else 0.0
        )

        return Transcription(
            text=text.strip(),
            language=info.language,
            confidence=avg_confidence,
            timestamp=datetime.now(),
            context_used=context
        )

# src/raywhisper/application/use_cases/transcribe_audio.py
from ...domain.interfaces.transcriber import ITranscriber
from ...domain.interfaces.vector_store import IVectorStore
from ...domain.value_objects.audio_data import AudioData
from ...domain.entities.transcription import Transcription

class TranscribeAudioUseCase:
    def __init__(
        self,
        transcriber: ITranscriber,
        vector_store: IVectorStore | None = None
    ):
        self._transcriber = transcriber
        self._vector_store = vector_store

    def execute(self, audio: AudioData, use_rag: bool = True) -> Transcription:
        context = None

        if use_rag and self._vector_store:
            # First pass: quick transcription for context retrieval
            initial_transcription = self._transcriber.transcribe(audio)

            # Retrieve relevant context
            results = self._vector_store.search(
                query=initial_transcription.text,
                top_k=5
            )
            context = "\n".join([r.content for r in results])

        # Final transcription with context
        return self._transcriber.transcribe(audio, context=context)
```

#### QA Checklist
- [ ] Whisper model loads successfully
- [ ] Transcription accuracy is high (manual testing)
- [ ] Transcription speed is acceptable (<2s for 10s audio)
- [ ] Context improves transcription quality (A/B testing)
- [ ] Memory usage is reasonable
- [ ] Unit tests cover all methods
- [ ] Integration tests with real audio files
- [ ] Performance benchmarks documented

---

### Phase 4: Embedding & Vector Database
**Status**: ⬜ Not Started

#### Tasks
- [ ] 4.1 Define IVectorStore interface
- [ ] 4.2 Create Document entity
- [ ] 4.3 Implement document parsers (Markdown, C#)
- [ ] 4.4 Implement embedding generator
- [ ] 4.5 Implement ChromaDB vector store
- [ ] 4.6 Create PopulateEmbeddings use case
- [ ] 4.7 Create CLI script for population
- [ ] 4.8 Write unit tests
- [ ] 4.9 Write integration tests

#### Pseudo-code: Vector Database

```python
# src/raywhisper/domain/interfaces/vector_store.py
from abc import ABC, abstractmethod
from typing import List
from ..entities.document import Document
from ..value_objects.context import SearchResult

class IVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store."""
        pass

# src/raywhisper/domain/entities/document.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Document:
    content: str
    source_path: Path
    metadata: dict

    @property
    def file_type(self) -> str:
        return self.source_path.suffix

# src/raywhisper/domain/value_objects/context.py
from dataclasses import dataclass

@dataclass(frozen=True)
class SearchResult:
    content: str
    score: float
    metadata: dict

# src/raywhisper/infrastructure/embeddings/document_parser.py
from pathlib import Path
from typing import List
from ...domain.entities.document import Document
import re

class DocumentParser:
    @staticmethod
    def parse_markdown(file_path: Path) -> List[Document]:
        """Parse markdown file into chunks."""
        content = file_path.read_text(encoding='utf-8')

        # Split by headers
        chunks = []
        current_chunk = []
        current_header = ""

        for line in content.split('\n'):
            if line.startswith('#'):
                if current_chunk:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'header': current_header
                    })
                current_header = line
                current_chunk = [line]
            else:
                current_chunk.append(line)

        if current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'header': current_header
            })

        return [
            Document(
                content=chunk['content'],
                source_path=file_path,
                metadata={'header': chunk['header'], 'type': 'markdown'}
            )
            for chunk in chunks if chunk['content'].strip()
        ]

    @staticmethod
    def parse_csharp(file_path: Path) -> List[Document]:
        """Parse C# file into chunks (classes, methods)."""
        content = file_path.read_text(encoding='utf-8')

        # Simple regex-based parsing (could use Roslyn for better parsing)
        chunks = []

        # Extract classes
        class_pattern = r'(public|private|protected|internal)?\s*(class|interface|struct)\s+(\w+).*?\{(.*?)\n\}'
        for match in re.finditer(class_pattern, content, re.DOTALL):
            chunks.append(Document(
                content=match.group(0),
                source_path=file_path,
                metadata={'type': 'csharp', 'kind': match.group(2), 'name': match.group(3)}
            ))

        return chunks

# src/raywhisper/infrastructure/vector_db/chroma_store.py
import chromadb
from chromadb.config import Settings
from typing import List
from ...domain.interfaces.vector_store import IVectorStore
from ...domain.entities.document import Document
from ...domain.value_objects.context import SearchResult

class ChromaVectorStore(IVectorStore):
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model_name: str
    ):
        self._client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Use sentence-transformers embedding function
        from chromadb.utils import embedding_functions
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn
        )

    def add_documents(self, documents: List[Document]) -> None:
        ids = [f"{doc.source_path}_{i}" for i, doc in enumerate(documents)]
        contents = [doc.content for doc in documents]
        metadatas = [
            {**doc.metadata, 'source': str(doc.source_path)}
            for doc in documents
        ]

        self._collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k
        )

        search_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                search_results.append(SearchResult(
                    content=doc,
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    metadata=results['metadatas'][0][i]
                ))

        return search_results

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)

# src/raywhisper/application/use_cases/populate_embeddings.py
from pathlib import Path
from typing import List
from ...domain.interfaces.vector_store import IVectorStore
from ...infrastructure.embeddings.document_parser import DocumentParser

class PopulateEmbeddingsUseCase:
    def __init__(self, vector_store: IVectorStore):
        self._vector_store = vector_store
        self._parser = DocumentParser()

    def execute(self, source_directories: List[Path], clear_existing: bool = False) -> int:
        if clear_existing:
            self._vector_store.clear()

        total_docs = 0

        for directory in source_directories:
            # Find all markdown and C# files
            md_files = list(directory.rglob("*.md"))
            cs_files = list(directory.rglob("*.cs"))

            for md_file in md_files:
                docs = self._parser.parse_markdown(md_file)
                self._vector_store.add_documents(docs)
                total_docs += len(docs)

            for cs_file in cs_files:
                docs = self._parser.parse_csharp(cs_file)
                self._vector_store.add_documents(docs)
                total_docs += len(docs)

        return total_docs
```

#### QA Checklist
- [ ] Document parsing works for markdown files
- [ ] Document parsing works for C# files
- [ ] Embeddings are generated correctly
- [ ] Vector search returns relevant results
- [ ] ChromaDB persists data correctly
- [ ] Population script handles large directories
- [ ] Unit tests cover all parsers
- [ ] Integration tests verify end-to-end flow

---

### Phase 5: RAG Retrieval with Reranking
**Status**: ⬜ Not Started

#### Tasks
- [ ] 5.1 Implement reranker
- [ ] 5.2 Create RAG service
- [ ] 5.3 Integrate with transcription
- [ ] 5.4 Optimize context window
- [ ] 5.5 Write unit tests
- [ ] 5.6 Write integration tests
- [ ] 5.7 A/B test with and without reranking

#### Pseudo-code: Reranking

```python
# src/raywhisper/infrastructure/embeddings/reranker.py
from sentence_transformers import CrossEncoder
from typing import List
from ...domain.value_objects.context import SearchResult

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int = 3
    ) -> List[SearchResult]:
        """Rerank search results using cross-encoder."""
        if not results:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, result.content) for result in results]

        # Get scores
        scores = self._model.predict(pairs)

        # Combine with original results and sort
        reranked = [
            SearchResult(
                content=result.content,
                score=float(score),
                metadata=result.metadata
            )
            for result, score in zip(results, scores)
        ]

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_n]

# src/raywhisper/application/services/rag_service.py
from typing import List
from ...domain.interfaces.vector_store import IVectorStore
from ...domain.value_objects.context import SearchResult
from ...infrastructure.embeddings.reranker import CrossEncoderReranker

class RAGService:
    def __init__(
        self,
        vector_store: IVectorStore,
        reranker: CrossEncoderReranker,
        top_k: int = 5,
        top_n: int = 3
    ):
        self._vector_store = vector_store
        self._reranker = reranker
        self._top_k = top_k
        self._top_n = top_n

    def retrieve_context(self, query: str) -> str:
        """Retrieve and rerank context for a query."""
        # Initial retrieval
        results = self._vector_store.search(query, top_k=self._top_k)

        if not results:
            return ""

        # Rerank
        reranked = self._reranker.rerank(query, results, top_n=self._top_n)

        # Format context
        context_parts = [
            f"[{r.metadata.get('source', 'unknown')}]\n{r.content}"
            for r in reranked
        ]

        return "\n\n---\n\n".join(context_parts)
```

#### QA Checklist
- [ ] Reranking improves result quality (manual evaluation)
- [ ] Reranking latency is acceptable (<500ms)
- [ ] Context window size is optimal
- [ ] RAG service integrates smoothly with transcription
- [ ] Unit tests cover reranking logic
- [ ] Integration tests verify full RAG pipeline
- [ ] A/B testing shows improvement

---

### Phase 6: Keyboard Control
**Status**: ⬜ Not Started

#### Tasks
- [ ] 6.1 Define IKeyboardController interface
- [ ] 6.2 Implement base keyboard controller
- [ ] 6.3 Implement Windows-specific controller
- [ ] 6.4 Implement macOS-specific controller
- [ ] 6.5 Create factory for platform detection
- [ ] 6.6 Add hotkey registration
- [ ] 6.7 Write unit tests (mocked)
- [ ] 6.8 Manual testing on Windows
- [ ] 6.9 Manual testing on macOS

#### Pseudo-code: Keyboard Control

```python
# src/raywhisper/domain/interfaces/keyboard_controller.py
from abc import ABC, abstractmethod
from typing import Callable

class IKeyboardController(ABC):
    @abstractmethod
    def register_hotkey(self, hotkey: str, callback: Callable[[], None]) -> None:
        """Register a global hotkey."""
        pass

    @abstractmethod
    def unregister_all(self) -> None:
        """Unregister all hotkeys."""
        pass

    @abstractmethod
    def start_listening(self) -> None:
        """Start listening for hotkeys."""
        pass

    @abstractmethod
    def stop_listening(self) -> None:
        """Stop listening for hotkeys."""
        pass

# src/raywhisper/infrastructure/keyboard/base_controller.py
from pynput import keyboard
from typing import Callable, Dict
from ...domain.interfaces.keyboard_controller import IKeyboardController

class PynputKeyboardController(IKeyboardController):
    def __init__(self):
        self._hotkeys: Dict[str, Callable] = {}
        self._listener = None

    def register_hotkey(self, hotkey: str, callback: Callable[[], None]) -> None:
        """Register hotkey in pynput format (e.g., '<ctrl>+<shift>+<space>')."""
        # Convert from our format to pynput format
        pynput_hotkey = self._convert_hotkey_format(hotkey)
        self._hotkeys[pynput_hotkey] = callback

    def _convert_hotkey_format(self, hotkey: str) -> str:
        """Convert 'ctrl+shift+space' to '<ctrl>+<shift>+<space>'."""
        parts = hotkey.lower().split('+')
        return '+'.join(f'<{part}>' for part in parts)

    def start_listening(self) -> None:
        if self._listener is None:
            self._listener = keyboard.GlobalHotKeys(self._hotkeys)
            self._listener.start()

    def stop_listening(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def unregister_all(self) -> None:
        self.stop_listening()
        self._hotkeys.clear()

# Platform-specific implementations if needed
# src/raywhisper/infrastructure/keyboard/windows_controller.py
# (Inherits from PynputKeyboardController, adds Windows-specific features if needed)

# src/raywhisper/infrastructure/keyboard/macos_controller.py
# (Inherits from PynputKeyboardController, adds macOS-specific features if needed)

# Factory
import platform

def create_keyboard_controller() -> IKeyboardController:
    system = platform.system()
    if system == "Windows":
        from .windows_controller import WindowsKeyboardController
        return WindowsKeyboardController()
    elif system == "Darwin":  # macOS
        from .macos_controller import MacOSKeyboardController
        return MacOSKeyboardController()
    else:
        raise NotImplementedError(f"Platform {system} not supported")
```

#### QA Checklist
- [ ] Hotkeys work on Windows
- [ ] Hotkeys work on macOS
- [ ] Hotkeys work globally (even when app not focused)
- [ ] Multiple hotkeys can be registered
- [ ] Hotkeys can be unregistered
- [ ] No conflicts with system hotkeys
- [ ] Manual testing checklist completed

---

### Phase 7: Output Simulation
**Status**: ⬜ Not Started

#### Tasks
- [ ] 7.1 Define IOutputSimulator interface
- [ ] 7.2 Implement keystroke simulator
- [ ] 7.3 Add typing speed control
- [ ] 7.4 Handle special characters
- [ ] 7.5 Add clipboard fallback (optional)
- [ ] 7.6 Write unit tests
- [ ] 7.7 Integration tests
- [ ] 7.8 Manual testing in various applications

#### Pseudo-code: Output Simulation

```python
# src/raywhisper/domain/interfaces/output_simulator.py
from abc import ABC, abstractmethod

class IOutputSimulator(ABC):
    @abstractmethod
    def type_text(self, text: str) -> None:
        """Simulate typing the given text."""
        pass

# src/raywhisper/infrastructure/output/keystroke_simulator.py
from pynput.keyboard import Controller, Key
import time
from ...domain.interfaces.output_simulator import IOutputSimulator

class KeystrokeSimulator(IOutputSimulator):
    def __init__(self, typing_speed: float = 0.01):
        self._keyboard = Controller()
        self._typing_speed = typing_speed

    def type_text(self, text: str) -> None:
        """Type text character by character."""
        for char in text:
            try:
                self._keyboard.type(char)
                time.sleep(self._typing_speed)
            except Exception as e:
                # Handle special characters that can't be typed
                print(f"Warning: Could not type character '{char}': {e}")
                continue

# Alternative with clipboard fallback
class SmartOutputSimulator(IOutputSimulator):
    def __init__(self, typing_speed: float = 0.01, use_clipboard_fallback: bool = False):
        self._keystroke_sim = KeystrokeSimulator(typing_speed)
        self._use_clipboard_fallback = use_clipboard_fallback

    def type_text(self, text: str) -> None:
        try:
            self._keystroke_sim.type_text(text)
        except Exception as e:
            if self._use_clipboard_fallback:
                self._fallback_to_clipboard(text)
            else:
                raise

    def _fallback_to_clipboard(self, text: str) -> None:
        """Fallback: copy to clipboard and paste."""
        import pyperclip
        pyperclip.copy(text)
        # Simulate Ctrl+V
        from pynput.keyboard import Controller, Key
        kb = Controller()
        with kb.pressed(Key.ctrl):
            kb.press('v')
            kb.release('v')
```

#### QA Checklist
- [ ] Text is typed correctly in various applications
- [ ] Special characters are handled
- [ ] Typing speed is configurable
- [ ] No characters are lost
- [ ] Works in text editors
- [ ] Works in IDEs
- [ ] Works in browsers
- [ ] Unit tests cover edge cases
- [ ] Integration tests verify output

---

### Phase 8: Integration & End-to-End Testing
**Status**: ⬜ Not Started

#### Tasks
- [ ] 8.1 Create main application orchestrator
- [ ] 8.2 Implement ProcessVoiceInput use case
- [ ] 8.3 Wire up all components with dependency injection
- [ ] 8.4 Create CLI interface
- [ ] 8.5 Add graceful shutdown
- [ ] 8.6 Write end-to-end tests
- [ ] 8.7 Performance optimization
- [ ] 8.8 Memory profiling
- [ ] 8.9 User acceptance testing

#### Pseudo-code: Main Application

```python
# src/raywhisper/application/use_cases/process_voice_input.py
from ...domain.interfaces.audio_recorder import IAudioRecorder
from ...domain.interfaces.transcriber import ITranscriber
from ...domain.interfaces.output_simulator import IOutputSimulator
from ..services.rag_service import RAGService
from loguru import logger

class ProcessVoiceInputUseCase:
    def __init__(
        self,
        audio_recorder: IAudioRecorder,
        transcriber: ITranscriber,
        rag_service: RAGService,
        output_simulator: IOutputSimulator
    ):
        self._audio_recorder = audio_recorder
        self._transcriber = transcriber
        self._rag_service = rag_service
        self._output_simulator = output_simulator

    def start_recording(self) -> None:
        """Start recording audio."""
        logger.info("Starting audio recording...")
        self._audio_recorder.start_recording()

    def stop_recording_and_transcribe(self) -> None:
        """Stop recording, transcribe, and output text."""
        logger.info("Stopping audio recording...")
        audio = self._audio_recorder.stop_recording()

        if audio.duration < 0.5:
            logger.warning("Audio too short, skipping transcription")
            return

        logger.info(f"Transcribing {audio.duration:.2f}s of audio...")

        # Get initial transcription for context retrieval
        initial_transcription = self._transcriber.transcribe(audio)
        logger.info(f"Initial transcription: {initial_transcription.text}")

        # Retrieve context
        context = self._rag_service.retrieve_context(initial_transcription.text)

        # Final transcription with context
        if context:
            logger.info("Re-transcribing with RAG context...")
            final_transcription = self._transcriber.transcribe(audio, context=context)
        else:
            final_transcription = initial_transcription

        logger.info(f"Final transcription: {final_transcription.text}")

        # Output text
        self._output_simulator.type_text(final_transcription.text)
        logger.info("Text output complete")

# src/raywhisper/presentation/app.py
from ..infrastructure.keyboard import create_keyboard_controller
from ..infrastructure.audio.sounddevice_recorder import SoundDeviceRecorder
from ..infrastructure.transcription.whisper_transcriber import WhisperTranscriber
from ..infrastructure.vector_db.chroma_store import ChromaVectorStore
from ..infrastructure.embeddings.reranker import CrossEncoderReranker
from ..infrastructure.output.keystroke_simulator import KeystrokeSimulator
from ..application.services.rag_service import RAGService
from ..application.use_cases.process_voice_input import ProcessVoiceInputUseCase
from ..config.settings import Settings
from loguru import logger
import sys

class RayWhisperApp:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._setup_logging()
        self._initialize_components()

    def _setup_logging(self):
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add("logs/raywhisper.log", rotation="10 MB", level="DEBUG")

    def _initialize_components(self):
        logger.info("Initializing RayWhisper components...")

        # Infrastructure
        self._audio_recorder = SoundDeviceRecorder(
            sample_rate=self._settings.audio.sample_rate,
            channels=self._settings.audio.channels
        )

        self._transcriber = WhisperTranscriber(
            model_size=self._settings.whisper.model_size,
            device=self._settings.whisper.device,
            compute_type=self._settings.whisper.compute_type
        )

        self._vector_store = ChromaVectorStore(
            collection_name=self._settings.vector_db.collection_name,
            persist_directory=self._settings.vector_db.persist_directory,
            embedding_model_name=self._settings.vector_db.embedding_model
        )

        self._reranker = CrossEncoderReranker(
            model_name=self._settings.reranker.model_name
        )

        self._output_simulator = KeystrokeSimulator(
            typing_speed=self._settings.output.typing_speed
        )

        self._keyboard_controller = create_keyboard_controller()

        # Application services
        self._rag_service = RAGService(
            vector_store=self._vector_store,
            reranker=self._reranker,
            top_k=self._settings.vector_db.top_k,
            top_n=self._settings.reranker.top_n
        )

        # Use cases
        self._voice_input_use_case = ProcessVoiceInputUseCase(
            audio_recorder=self._audio_recorder,
            transcriber=self._transcriber,
            rag_service=self._rag_service,
            output_simulator=self._output_simulator
        )

        logger.info("Components initialized successfully")

    def run(self):
        """Start the application."""
        logger.info("Starting RayWhisper...")

        # Register hotkeys
        self._keyboard_controller.register_hotkey(
            self._settings.keyboard.start_stop_hotkey,
            self._toggle_recording
        )

        self._keyboard_controller.start_listening()

        logger.info(f"Listening for hotkey: {self._settings.keyboard.start_stop_hotkey}")
        logger.info("Press Ctrl+C to exit")

        try:
            # Keep running
            import threading
            event = threading.Event()
            event.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.shutdown()

    def _toggle_recording(self):
        """Toggle between start and stop recording."""
        if self._audio_recorder.is_recording():
            self._voice_input_use_case.stop_recording_and_transcribe()
        else:
            self._voice_input_use_case.start_recording()

    def shutdown(self):
        """Graceful shutdown."""
        self._keyboard_controller.stop_listening()
        logger.info("RayWhisper stopped")

# src/raywhisper/presentation/cli.py
import click
from .app import RayWhisperApp
from ..config.settings import Settings

@click.group()
def cli():
    """RayWhisper - Voice to Text with RAG"""
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Path to config file')
def run(config):
    """Run the voice-to-text application."""
    settings = Settings()
    app = RayWhisperApp(settings)
    app.run()

@cli.command()
@click.argument('directories', nargs=-1, type=click.Path(exists=True))
@click.option('--clear', is_flag=True, help='Clear existing embeddings')
def populate(directories, clear):
    """Populate the embedding database from directories."""
    from pathlib import Path
    from ..infrastructure.vector_db.chroma_store import ChromaVectorStore
    from ..application.use_cases.populate_embeddings import PopulateEmbeddingsUseCase

    settings = Settings()
    vector_store = ChromaVectorStore(
        collection_name=settings.vector_db.collection_name,
        persist_directory=settings.vector_db.persist_directory,
        embedding_model_name=settings.vector_db.embedding_model
    )

    use_case = PopulateEmbeddingsUseCase(vector_store)
    paths = [Path(d) for d in directories]

    click.echo(f"Populating embeddings from {len(paths)} directories...")
    total = use_case.execute(paths, clear_existing=clear)
    click.echo(f"Successfully added {total} documents to the vector database")

if __name__ == '__main__':
    cli()
```

#### QA Checklist
- [ ] All components integrate correctly
- [ ] Dependency injection works properly
- [ ] Application starts without errors
- [ ] Hotkeys trigger recording
- [ ] Full workflow works end-to-end
- [ ] Graceful shutdown works
- [ ] Memory usage is acceptable
- [ ] CPU usage is reasonable
- [ ] No memory leaks
- [ ] Performance meets requirements
- [ ] End-to-end tests pass
- [ ] User acceptance testing completed

---

## QA Strategy

### Testing Pyramid

```
        ┌─────────────┐
        │   E2E (5%)  │  Full workflow tests
        ├─────────────┤
        │ Integration │  Component interaction tests
        │    (25%)    │
        ├─────────────┤
        │    Unit     │  Individual component tests
        │    (70%)    │
        └─────────────┘
```

### Unit Testing Strategy

**Framework**: pytest with pytest-cov for coverage

**Coverage Target**: >80% for all modules

**Key Areas**:
- Domain entities and value objects (100% coverage)
- Use cases (>90% coverage)
- Infrastructure adapters (>70% coverage)

**Example Test Structure**:
```python
# tests/unit/domain/entities/test_transcription.py
import pytest
from raywhisper.domain.entities.transcription import Transcription
from datetime import datetime

def test_transcription_creation():
    trans = Transcription(
        text="Hello world",
        language="en",
        confidence=0.95,
        timestamp=datetime.now()
    )
    assert trans.text == "Hello world"
    assert trans.confidence == 0.95

def test_transcription_invalid_confidence():
    with pytest.raises(ValueError):
        Transcription(
            text="Hello",
            language="en",
            confidence=1.5,  # Invalid
            timestamp=datetime.now()
        )
```

### Integration Testing Strategy

**Focus**: Component interactions and external dependencies

**Key Scenarios**:
1. Audio recording → Transcription
2. Document parsing → Vector DB
3. Vector DB → Reranking
4. Full RAG pipeline

**Example**:
```python
# tests/integration/test_rag_pipeline.py
import pytest
from pathlib import Path

def test_full_rag_pipeline(vector_store, reranker, rag_service):
    # Populate with test documents
    test_docs = [
        Document(content="Python is a programming language", ...),
        Document(content="C# is used for .NET development", ...)
    ]
    vector_store.add_documents(test_docs)

    # Retrieve context
    context = rag_service.retrieve_context("programming languages")

    assert "Python" in context or "C#" in context
    assert len(context) > 0
```

### End-to-End Testing Strategy

**Scenarios**:
1. Complete voice-to-text workflow
2. Embedding population workflow
3. Error recovery scenarios

**Manual Testing Checklist**:
- [ ] Record 5-second audio → verify transcription
- [ ] Record 30-second audio → verify transcription
- [ ] Test with background noise
- [ ] Test with different accents
- [ ] Test technical terminology (with RAG)
- [ ] Test in different applications (VS Code, Word, Browser)
- [ ] Test hotkey conflicts
- [ ] Test rapid start/stop toggling

### Performance Testing

**Benchmarks**:
- Whisper transcription: <2s for 10s audio (base model)
- Vector search: <100ms for top-5 retrieval
- Reranking: <500ms for top-3
- End-to-end latency: <3s for 10s audio

**Tools**: pytest-benchmark, memory_profiler

### Platform-Specific Testing

**Windows Testing**:
- [ ] Windows 10
- [ ] Windows 11
- [ ] Different keyboard layouts
- [ ] Different audio devices

**macOS Testing**:
- [ ] macOS Monterey
- [ ] macOS Ventura
- [ ] macOS Sonoma
- [ ] Accessibility permissions

---

## Configuration Specifications

### config.yaml Structure

```yaml
# Audio Configuration
audio:
  sample_rate: 16000        # Hz
  channels: 1               # Mono
  chunk_duration: 0.1       # seconds

# Whisper Model Configuration
whisper:
  model_size: "base"        # tiny, base, small, medium, large-v2, large-v3
  device: "auto"            # cpu, cuda, auto
  compute_type: "float16"   # int8, float16, float32
  language: null            # null for auto-detect, or "en", "es", etc.

# Vector Database Configuration
vector_db:
  collection_name: "raywhisper_docs"
  embedding_model: "all-MiniLM-L6-v2"
  persist_directory: "./data/chroma"
  top_k: 5                  # Number of results to retrieve

# Reranker Configuration
reranker:
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_n: 3                  # Number of results after reranking

# Keyboard Configuration
keyboard:
  start_stop_hotkey: "ctrl+shift+space"
  cancel_hotkey: "ctrl+shift+esc"

# Output Configuration
output:
  typing_speed: 0.01        # Delay between keystrokes (seconds)
  use_clipboard_fallback: false
```

### Environment Variables

```bash
# .env
RAYWHISPER_WHISPER__MODEL_SIZE=base
RAYWHISPER_WHISPER__DEVICE=cuda
RAYWHISPER_VECTOR_DB__PERSIST_DIRECTORY=/path/to/data
RAYWHISPER_KEYBOARD__START_STOP_HOTKEY=ctrl+shift+space
```

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Whisper model too slow | High | Medium | Use faster-whisper with quantization; offer model size options |
| Keyboard shortcuts conflict with system | Medium | High | Make hotkeys configurable; detect conflicts |
| Keystroke simulation unreliable | High | Medium | Extensive testing; clipboard fallback option |
| Cross-platform compatibility issues | High | Medium | Abstract platform-specific code; test on both platforms early |
| Vector DB performance degradation | Medium | Low | Monitor performance; implement caching; limit DB size |
| Audio quality issues | High | Medium | Implement audio preprocessing; noise reduction |
| Memory leaks in long-running process | Medium | Medium | Implement proper cleanup; memory profiling |

### Mitigation Strategies

**1. Model Performance**
- Start with base model, allow users to choose
- Implement model caching to avoid reload
- Consider using distilled models for speed

**2. Cross-Platform Issues**
- Use pynput for maximum compatibility
- Abstract platform-specific code behind interfaces
- Test on both platforms from Phase 2 onwards

**3. Keyboard Reliability**
- Implement retry logic for failed keystrokes
- Add optional clipboard fallback
- Log all failures for debugging

**4. Audio Quality**
- Implement VAD (Voice Activity Detection)
- Add audio preprocessing (noise reduction)
- Allow users to test audio input

**5. RAG Quality**
- A/B test with and without RAG
- Allow users to disable RAG if not helpful
- Implement feedback mechanism

---

## Progress Tracking

### Overall Progress

- [ ] **Phase 1**: Project Setup & Core Infrastructure (0/6 tasks)
- [ ] **Phase 2**: Audio Recording Module (0/7 tasks)
- [ ] **Phase 3**: Transcription Module (0/9 tasks)
- [ ] **Phase 4**: Embedding & Vector Database (0/9 tasks)
- [ ] **Phase 5**: RAG Retrieval with Reranking (0/7 tasks)
- [ ] **Phase 6**: Keyboard Control (0/9 tasks)
- [ ] **Phase 7**: Output Simulation (0/8 tasks)
- [ ] **Phase 8**: Integration & E2E Testing (0/9 tasks)

### Milestones

- [ ] **M1**: Basic transcription working (Phases 1-3)
- [ ] **M2**: RAG pipeline functional (Phases 4-5)
- [ ] **M3**: Keyboard control working (Phase 6)
- [ ] **M4**: Full integration complete (Phases 7-8)
- [ ] **M5**: Production ready (All QA passed)

### Definition of Done (per Phase)

A phase is considered complete when:
1. ✅ All tasks in the phase are completed
2. ✅ All unit tests pass with >80% coverage
3. ✅ All integration tests pass
4. ✅ QA checklist is fully checked
5. ✅ Code review completed (if applicable)
6. ✅ Documentation updated
7. ✅ No critical bugs remain

### Estimated Timeline

| Phase | Estimated Duration | Dependencies |
|-------|-------------------|--------------|
| Phase 1 | 1-2 days | None |
| Phase 2 | 2-3 days | Phase 1 |
| Phase 3 | 3-4 days | Phase 1, 2 |
| Phase 4 | 3-4 days | Phase 1 |
| Phase 5 | 2-3 days | Phase 3, 4 |
| Phase 6 | 2-3 days | Phase 1 |
| Phase 7 | 2-3 days | Phase 1 |
| Phase 8 | 3-5 days | All previous |
| **Total** | **18-27 days** | |

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Set up development environment** (Python, IDE, dependencies)
3. **Begin Phase 1** - Project setup
4. **Establish CI/CD pipeline** (optional but recommended)
5. **Create project repository** structure
6. **Start implementation** following the phases

---

## Appendix

### Useful Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=src/raywhisper

# Run linting
ruff check src/
mypy src/

# Run the application
python -m raywhisper.presentation.cli run

# Populate embeddings
python -m raywhisper.presentation.cli populate ./docs ./src --clear

# Run specific test
pytest tests/unit/domain/test_transcription.py -v
```

### References

- [faster-whisper Documentation](https://github.com/guillaumekln/faster-whisper)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [sentence-transformers Documentation](https://www.sbert.net/)
- [pynput Documentation](https://pynput.readthedocs.io/)
- [Clean Architecture Principles](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-24
**Status**: Ready for Implementation


