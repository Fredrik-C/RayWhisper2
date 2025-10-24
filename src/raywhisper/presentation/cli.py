"""Command-line interface for RayWhisper."""

from pathlib import Path

import click
from loguru import logger

from ..application.use_cases.populate_embeddings import PopulateEmbeddingsUseCase
from ..config.loader import load_settings
from ..config.settings import Settings
from ..infrastructure.vector_db.chroma_store import ChromaVectorStore
from .app import RayWhisperApp


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """RayWhisper - Voice to Text with RAG-enhanced transcription."""
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to config file (optional)",
)
def run(config: str | None) -> None:
    """Run the voice-to-text application.

    Hold the configured key combination (default: Ctrl+Shift+Space) to record.
    Release the keys to stop recording and transcribe.
    The transcribed text will be typed into the active application.
    """
    try:
        settings = load_settings(config)
        logger.info("Starting RayWhisper application")
        app = RayWhisperApp(settings)
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command()
@click.argument("directories", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--clear", is_flag=True, help="Clear existing embeddings before populating")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to config file (optional)",
)
def populate(directories: tuple[str, ...], clear: bool, config: str | None) -> None:
    """Populate the embedding database from directories.

    Scans the specified directories for Markdown (.md) and C# (.cs) files,
    parses them into chunks, and adds them to the vector database.

    Example:
        raywhisper populate ./docs ./src --clear
    """
    try:
        settings = load_settings(config)

        click.echo(f"Initializing vector store at {settings.vector_db.persist_directory}")
        vector_store = ChromaVectorStore(
            collection_name=settings.vector_db.collection_name,
            persist_directory=settings.vector_db.persist_directory,
            embedding_model_name=settings.vector_db.embedding_model,
            chunk_size=settings.vector_db.chunk_size,
            chunk_overlap=settings.vector_db.chunk_overlap,
            use_query_instruction=settings.vector_db.use_query_instruction,
        )

        use_case = PopulateEmbeddingsUseCase(
            vector_store,
            chunk_size=settings.vector_db.chunk_size,
            chunk_overlap=settings.vector_db.chunk_overlap,
        )
        paths = [Path(d) for d in directories]

        if clear:
            click.echo("Clearing existing embeddings...")

        click.echo(f"Populating embeddings from {len(paths)} directories...")

        with click.progressbar(
            length=len(paths),
            label="Processing directories",
        ) as bar:
            total = use_case.execute(paths, clear_existing=clear)
            bar.update(len(paths))

        click.echo(f"✓ Successfully added {total} documents to the vector database")

    except Exception as e:
        logger.error(f"Failed to populate embeddings: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to config file (optional)",
)
def info(config: str | None) -> None:
    """Display information about the current configuration."""
    try:
        settings = load_settings(config)

        click.echo("RayWhisper Configuration:")
        click.echo(f"  Whisper Model: {settings.whisper.model_size}")
        click.echo(f"  Device: {settings.whisper.device}")
        click.echo(f"  Compute Type: {settings.whisper.compute_type}")
        click.echo(f"  Embedding Model: {settings.vector_db.embedding_model}")
        click.echo(f"  Chunk Size: {settings.vector_db.chunk_size} words")
        click.echo(f"  Chunk Overlap: {settings.vector_db.chunk_overlap} words")
        click.echo(f"  Query Instruction: {settings.vector_db.use_query_instruction}")
        click.echo(f"  Reranker Model: {settings.reranker.model_name}")
        click.echo(f"  Vector DB: {settings.vector_db.persist_directory}")
        click.echo(f"  Hotkey: {settings.keyboard.start_stop_hotkey}")

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        raise click.ClickException(str(e))


if __name__ == "__main__":
    cli()

