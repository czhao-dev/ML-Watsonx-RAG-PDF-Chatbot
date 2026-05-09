"""Run the Gradio app."""

from app.config import get_settings
from app.ui import build_interface


def main() -> None:
    settings = get_settings()
    app = build_interface()
    app.launch(server_name=settings.server_name, server_port=settings.server_port)


if __name__ == "__main__":
    main()
