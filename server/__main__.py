import os
import uvicorn


def main() -> None:
    host = os.getenv("HOME_AUTOMATION_BIND", "0.0.0.0")
    port = int(os.getenv("HOME_AUTOMATION_PORT", "8123"))
    uvicorn.run("server.main:app", host=host, port=port, reload=False, log_level=os.getenv("HOME_AUTOMATION_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()

