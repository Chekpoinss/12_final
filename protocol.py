import json
import socket
from typing import Any, Dict


class JsonSocket:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.file = sock.makefile("r", encoding="utf-8", newline="\n")

    def send(self, payload: Dict[str, Any]) -> None:
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.sock.sendall(data)

    def recv(self) -> Dict[str, Any]:
        line = self.file.readline()
        if not line:
            raise ConnectionError("Socket closed by peer")
        return json.loads(line)

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            self.sock.close()