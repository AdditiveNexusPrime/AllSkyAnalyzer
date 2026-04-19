"""
AllSkyAnalyzer - Night Sky Image Analysis Application

Entry point for the AllSkyAnalyzer application. Supports running as a
one-shot analysis, a continuous watcher, or a REST API server.
"""

import sys
import logging

import click
import yaml

from src.config import Config
from src.analyzer import AllSkyAnalyzer
from src.watcher import ImageWatcher
from src.api import create_app

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@click.group()
@click.option(
    "--config",
    "-c",
    default="config/config.yaml",
    show_default=True,
    help="Path to YAML configuration file.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity.",
)
@click.pass_context
def cli(ctx: click.Context, config: str, log_level: str) -> None:
    """AllSkyAnalyzer – analyze allsky camera images for astronomical objects and anomalies."""
    _setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.from_yaml(config)


@cli.command()
@click.argument("image_path")
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Directory to write overlay images. Defaults to config value.",
)
@click.pass_context
def analyze(ctx: click.Context, image_path: str, output_dir: str | None) -> None:
    """Analyze a single allsky IMAGE_PATH and write results."""
    cfg: Config = ctx.obj["config"]
    if output_dir:
        cfg.output.directory = output_dir

    analyzer = AllSkyAnalyzer(cfg)
    result = analyzer.analyze_image(image_path)

    click.echo(f"Stars detected  : {len(result.stars)}")
    click.echo(f"Anomalies found : {len(result.anomalies)}")
    click.echo(f"Overlay saved   : {result.overlay_path}")


@cli.command()
@click.pass_context
def watch(ctx: click.Context) -> None:
    """Watch the indi-allsky image directory and analyze new images as they arrive."""
    cfg: Config = ctx.obj["config"]
    watcher = ImageWatcher(cfg)
    click.echo(f"Watching {cfg.indi_allsky.image_dir} for new images… (Ctrl-C to stop)")
    try:
        watcher.run()
    except KeyboardInterrupt:
        click.echo("Stopped.")


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host.")
@click.option("--port", default=8000, show_default=True, help="Bind port.")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the REST API server."""
    import uvicorn

    cfg: Config = ctx.obj["config"]
    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli(obj={})
