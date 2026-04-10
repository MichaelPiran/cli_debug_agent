import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groq import Groq

try:
    from utis import get_groq_api_key
except ImportError:
    get_groq_api_key = None


SYSTEM_PROMPT = """
You are a CLI debugging agent powered by Groq.

Your job:
- understand command line errors and runtime failures
- explain the most likely root cause in plain Italian
- suggest practical fixes, ordered from most likely to least likely
- use workspace inspection tools when they help confirm the diagnosis
- avoid inventing facts that are not supported by command output or tool results
- be concise and practical

Response format:
1. Problema
2. Causa probabile
3. Soluzioni consigliate (max 3)
4. Prossimo comando utile

Style rules:
- keep the full response under 140 words when possible
- prefer short sentences
- do not repeat the error text verbatim unless necessary
""".strip()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
# DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_MAX_STEPS = 6
DEFAULT_OUTPUT_LIMIT = 12000
DEFAULT_MAX_COMPLETION_TOKENS = 350
LAST_CAPTURE_FILENAME = ".last_debug_capture.json"
BASH_HISTORY_CANDIDATES = (".bash_history", ".zhistory")
REDACTED_PATH = "<PATH>"
REDACTED_HOME = "<HOME>"
REDACTED_WORKSPACE = "<WORKSPACE>"
REDACTED_API_KEY = "<API_KEY>"
LOG_GLOB_PATTERNS = (
    "*.log",
    "*.err",
    "*.out",
    "nohup.out",
    "logs/**/*.log",
    "**/*.log",
    "**/*.err",
    "**/*.out",
)


def _safe_path(workspace: Path, relative_path: str) -> Path:
    candidate = (workspace / relative_path).resolve()
    workspace_root = workspace.resolve()
    if candidate != workspace_root and workspace_root not in candidate.parents:
        raise ValueError("Path fuori dalla workspace")
    return candidate


def _truncate(text: str, limit: int = DEFAULT_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n...[troncato, {omitted} caratteri omessi]"


def _read_text_tail(path: Path, limit: int = DEFAULT_OUTPUT_LIMIT) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _extract_latest_error_block(text: str) -> str | None:
    traceback_marker = "Traceback (most recent call last):"
    last_traceback = text.rfind(traceback_marker)
    if last_traceback != -1:
        return text[last_traceback:].strip()

    lines = text.splitlines()
    if not lines:
        return None

    error_pattern = re.compile(
        r"(exception|error|fatal|failed|panic|cannot |can't |no such file|not found|permission denied|module not found|command not found)",
        re.IGNORECASE,
    )
    for index in range(len(lines) - 1, -1, -1):
        if error_pattern.search(lines[index]):
            start = max(0, index - 20)
            end = min(len(lines), index + 8)
            block = "\n".join(lines[start:end]).strip()
            return block or None

    return None


def sanitize_text(text: str, workspace: Path | None = None, cwd: str | None = None) -> str:
    sanitized = text

    replacement_pairs = []
    home = str(Path.home().resolve())
    replacement_pairs.append((home, REDACTED_HOME))

    if workspace is not None:
        replacement_pairs.append((str(workspace.resolve()), REDACTED_WORKSPACE))

    if cwd:
        replacement_pairs.append((cwd, "<CWD>"))

    replacement_pairs.sort(key=lambda item: len(item[0]), reverse=True)
    for old_value, new_value in replacement_pairs:
        if old_value:
            sanitized = sanitized.replace(old_value, new_value)

    sensitive_patterns = [
        (r"gsk_[A-Za-z0-9_\-]+", REDACTED_API_KEY),
        (r"\bsk-[A-Za-z0-9_\-]+\b", REDACTED_API_KEY),
        (r"\bAKIA[0-9A-Z]{16}\b", REDACTED_API_KEY),
        (r"(?<![A-Za-z0-9_./-])/(?:[^\s/:]+/)*[^\s:]+", REDACTED_PATH),
    ]
    for pattern, replacement in sensitive_patterns:
        sanitized = re.sub(pattern, replacement, sanitized)

    return sanitized


def resolve_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key

    if get_groq_api_key is not None:
        api_key = get_groq_api_key()
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            return api_key

    raise RuntimeError("Imposta GROQ_API_KEY oppure fornisci una funzione get_groq_api_key() valida")


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def as_groq_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class CommandCapture:
    command: str
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    source: str = "command"

    def as_prompt_block(self) -> str:
        return json.dumps(
            {
                "source": self.source,
                "command": self.command,
                "cwd": self.cwd,
                "exit_code": self.exit_code,
                "stdout": _truncate(self.stdout),
                "stderr": _truncate(self.stderr),
            },
            ensure_ascii=True,
            indent=2,
        )

    def as_model_payload(self, workspace: Path, allow_workspace_context: bool) -> str:
        error_text = self.stderr.strip() or self.stdout.strip()
        error_block = _extract_latest_error_block(error_text) or error_text
        if not error_block:
            error_block = "Nessun output di errore disponibile"

        payload: dict[str, Any] = {
            "source": self.source,
            "exit_code": self.exit_code,
            "command": sanitize_text(self.command, workspace=workspace, cwd=self.cwd),
            "error": sanitize_text(_truncate(error_block), workspace=workspace, cwd=self.cwd),
        }
        if allow_workspace_context:
            payload["workspace_context"] = "enabled"

        return json.dumps(payload, ensure_ascii=True, indent=2)

    def to_json(self) -> str:
        return json.dumps(
            {
                "source": self.source,
                "command": self.command,
                "cwd": self.cwd,
                "exit_code": self.exit_code,
                "stdout": self.stdout,
                "stderr": self.stderr,
            },
            ensure_ascii=True,
            indent=2,
        )

    @classmethod
    def from_json(cls, content: str) -> "CommandCapture":
        data = json.loads(content)
        return cls(
            source=data.get("source", "command"),
            command=data["command"],
            cwd=data["cwd"],
            exit_code=data["exit_code"],
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
        )


def get_last_capture_path(workspace: Path) -> Path:
    return workspace.resolve() / "cli_debug_agent" / LAST_CAPTURE_FILENAME


def save_capture(workspace: Path, capture: CommandCapture) -> None:
    cache_path = get_last_capture_path(workspace)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(capture.to_json(), encoding="utf-8")


def load_saved_capture(workspace: Path) -> CommandCapture | None:
    cache_path = get_last_capture_path(workspace)
    if not cache_path.exists():
        return None
    return CommandCapture.from_json(cache_path.read_text(encoding="utf-8"))


def _resolve_history_file(explicit_history_file: str | None = None) -> Path | None:
    if explicit_history_file:
        path = Path(explicit_history_file).expanduser()
        return path if path.exists() else None

    env_history = os.environ.get("HISTFILE")
    if env_history:
        path = Path(env_history).expanduser()
        if path.exists():
            return path

    home = Path.home()
    for candidate in BASH_HISTORY_CANDIDATES:
        path = home / candidate
        if path.exists():
            return path

    return None


def _is_debug_agent_command(command: str) -> bool:
    lowered = command.casefold()
    return "debug_agent.py" in lowered or "cli_debug_agent" in lowered


def find_last_shell_command(explicit_history_file: str | None = None) -> str | None:
    history_file = _resolve_history_file(explicit_history_file)
    if history_file is None:
        return None

    try:
        lines = history_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    for raw_line in reversed(lines):
        command = raw_line.strip()
        if not command:
            continue
        if command.startswith(": ") and ";" in command:
            command = command.split(";", 1)[1].strip()
        if not command or _is_debug_agent_command(command):
            continue
        return command

    return None


def replay_last_shell_command(cwd: Path, explicit_history_file: str | None = None) -> CommandCapture | None:
    command = find_last_shell_command(explicit_history_file)
    if command is None:
        return None

    capture = run_command(command, cwd=cwd, use_shell=True)
    capture.source = "history_replay"
    return capture


def find_recent_log_capture(workspace: Path, explicit_log_file: str | None = None) -> CommandCapture | None:
    candidate_files: list[Path] = []

    if explicit_log_file:
        log_path = _safe_path(workspace, explicit_log_file)
        if log_path.is_file():
            candidate_files.append(log_path)
    else:
        seen: set[Path] = set()
        for pattern in LOG_GLOB_PATTERNS:
            for path in workspace.resolve().glob(pattern):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    candidate_files.append(path)

        candidate_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        candidate_files = candidate_files[:30]

    for path in candidate_files:
        try:
            tail = _read_text_tail(path)
        except OSError:
            continue

        error_block = _extract_latest_error_block(tail)
        if not error_block:
            continue

        return CommandCapture(
            source="log_file",
            command=f"log:{path.relative_to(workspace.resolve()).as_posix()}",
            cwd=str(workspace.resolve()),
            exit_code=1,
            stdout="",
            stderr=error_block,
        )

    return None


class WorkspaceTools:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def list_files(self, path: str = ".") -> str:
        target = _safe_path(self.workspace, path)
        if not target.exists():
            raise FileNotFoundError(f"Percorso non trovato: {path}")
        if not target.is_dir():
            raise ValueError(f"Il percorso non e' una directory: {path}")

        entries = []
        for child in sorted(target.iterdir()):
            relative_name = child.relative_to(self.workspace).as_posix()
            entries.append(relative_name + ("/" if child.is_dir() else ""))

        return json.dumps({"path": path, "entries": entries}, ensure_ascii=True)

    def read_file(self, path: str, start_line: int = 1, end_line: int = 250) -> str:
        target = _safe_path(self.workspace, path)
        if not target.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        if not target.is_file():
            raise ValueError(f"Il percorso non e' un file: {path}")

        with target.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()

        selected = lines[start_line - 1 : end_line]
        return json.dumps(
            {
                "path": path,
                "start_line": start_line,
                "end_line": min(end_line, len(lines)),
                "content": "".join(selected),
            },
            ensure_ascii=True,
        )

    def search_text(self, query: str, path: str = ".", max_results: int = 20) -> str:
        target = _safe_path(self.workspace, path)
        if not target.exists():
            raise FileNotFoundError(f"Percorso non trovato: {path}")

        matches: list[dict[str, Any]] = []
        files = [target] if target.is_file() else [item for item in target.rglob("*") if item.is_file()]
        lowered = query.casefold()

        for file_path in sorted(files):
            try:
                with file_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if lowered in line.casefold():
                            matches.append(
                                {
                                    "path": file_path.relative_to(self.workspace).as_posix(),
                                    "line": line_number,
                                    "text": line.rstrip(),
                                }
                            )
                            if len(matches) >= max_results:
                                return json.dumps({"query": query, "matches": matches}, ensure_ascii=True)
            except UnicodeDecodeError:
                continue

        return json.dumps({"query": query, "matches": matches}, ensure_ascii=True)


class DebugAgent:
    def __init__(self, workspace: Path, model: str) -> None:
        self.client = Groq(api_key=resolve_api_key())
        self.model = model
        self.workspace_tools = WorkspaceTools(workspace)
        self.tool_specs = [
            ToolSpec(
                name="list_files",
                description="Elenca file e directory nella workspace o in una sottodirectory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Percorso relativo alla workspace", "default": "."}
                    },
                },
            ),
            ToolSpec(
                name="read_file",
                description="Legge righe di un file della workspace per verificare stacktrace, import o configurazioni.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Percorso relativo del file"},
                        "start_line": {"type": "integer", "default": 1},
                        "end_line": {"type": "integer", "default": 250},
                    },
                    "required": ["path"],
                },
            ),
            ToolSpec(
                name="search_text",
                description="Cerca testo nella workspace per trovare riferimenti a errori, simboli, import o configurazioni.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Testo da cercare"},
                        "path": {"type": "string", "default": "."},
                        "max_results": {"type": "integer", "default": 20},
                    },
                    "required": ["query"],
                },
            ),
        ]
        self.tool_map = {
            "list_files": self.workspace_tools.list_files,
            "read_file": self.workspace_tools.read_file,
            "search_text": self.workspace_tools.search_text,
        }
        self.workspace = workspace.resolve()

    def analyze(
        self,
        capture: CommandCapture,
        max_steps: int,
        allow_workspace_context: bool,
        print_prompt: bool,
    ) -> str:
        if allow_workspace_context:
            user_prompt = (
                "Analizza questo errore di terminale. I dati sensibili sono gia' stati sanitizzati. "
                "Se serve, usa i tool per ispezionare i file della workspace e confermare la diagnosi.\n\n"
                f"Dettagli esecuzione:\n{capture.as_model_payload(self.workspace, allow_workspace_context=True)}"
            )
        else:
            user_prompt = (
                "Analizza questo errore di terminale usando solo il payload sanitizzato seguente. "
                "Non hai accesso alla workspace e devi evitare assunzioni non supportate dai dati.\n\n"
                f"Dettagli esecuzione:\n{capture.as_model_payload(self.workspace, allow_workspace_context=False)}"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if print_prompt:
            preview_request: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            }
            if allow_workspace_context:
                preview_request["tools"] = [spec.as_groq_tool() for spec in self.tool_specs]
                preview_request["tool_choice"] = "auto"

            print("=== MODEL REQUEST BEGIN ===", file=sys.stderr)
            print(json.dumps(preview_request, ensure_ascii=True, indent=2), file=sys.stderr)
            print("=== MODEL REQUEST END ===", file=sys.stderr)

        for _ in range(max_steps):
            request: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            }
            if allow_workspace_context:
                request["tools"] = [spec.as_groq_tool() for spec in self.tool_specs]
                request["tool_choice"] = "auto"

            completion = self.client.chat.completions.create(**request)

            message = completion.choices[0].message
            assistant_payload: dict[str, Any] = {"role": "assistant"}
            if message.content:
                assistant_payload["content"] = message.content
            if message.tool_calls:
                assistant_payload["tool_calls"] = [tool_call.model_dump() for tool_call in message.tool_calls]
            messages.append(assistant_payload)

            if not message.tool_calls:
                return message.content or "Nessuna diagnosi generata"

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments or "{}"

                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    content = json.dumps({"error": f"Argomenti JSON non validi: {exc}"}, ensure_ascii=True)
                else:
                    handler = self.tool_map.get(tool_name)
                    if handler is None:
                        content = json.dumps({"error": f"Tool non supportato: {tool_name}"}, ensure_ascii=True)
                    else:
                        try:
                            content = handler(**arguments)
                        except Exception as exc:
                            content = json.dumps({"error": f"Errore durante {tool_name}: {exc}"}, ensure_ascii=True)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": sanitize_text(content, workspace=self.workspace),
                    }
                )

        raise RuntimeError(f"Numero massimo di step raggiunto ({max_steps})")


def run_command(command: str, cwd: Path, use_shell: bool) -> CommandCapture:
    completed = subprocess.run(
        command if use_shell else shlex.split(command),
        cwd=str(cwd),
        shell=use_shell,
        capture_output=True,
        text=True,
    )
    return CommandCapture(
        command=command,
        cwd=str(cwd.resolve()),
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        source="command",
    )


def read_stdin_capture(cwd: Path) -> CommandCapture:
    content = sys.stdin.read()
    if not content.strip():
        raise RuntimeError("Nessun input ricevuto da stdin")
    return CommandCapture(
        command="stdin",
        cwd=str(cwd.resolve()),
        exit_code=1,
        stdout="",
        stderr=content,
        source="stdin",
    )


def read_capture_file(file_path: str, cwd: Path, command: str | None, exit_code: int) -> CommandCapture:
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File di cattura non trovato: {file_path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    if not content.strip():
        raise RuntimeError("Il file di cattura e' vuoto")

    return CommandCapture(
        command=command or f"capture:{path}",
        cwd=str(cwd.resolve()),
        exit_code=exit_code,
        stdout="",
        stderr=content,
        source="capture_file",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente CLI di debugging con Groq")
    parser.add_argument(
        "--command",
        help="Comando da eseguire e analizzare. Esempio: --command 'python app.py'",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Esegue il comando tramite shell. Utile per pipe, redirect e comandi composti.",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace visibile all'agente. Default: directory corrente",
    )
    parser.add_argument(
        "--log-file",
        help="File di log da cui leggere automaticamente l'ultimo traceback o errore",
    )
    parser.add_argument(
        "--from-file",
        help="File contenente output di terminale gia' catturato da analizzare",
    )
    parser.add_argument(
        "--captured-command",
        help="Comando originale associato al contenuto passato con --from-file",
    )
    parser.add_argument(
        "--captured-exit-code",
        type=int,
        default=1,
        help="Exit code del comando associato a --from-file. Default: 1",
    )
    parser.add_argument(
        "--allow-workspace-context",
        action="store_true",
        help="Permette al modello di usare tool locali della workspace. Di default vengono inviati solo errori sanitizzati.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Stampa su stderr la request sanitizzata inviata al modello prima della chiamata API.",
    )
    parser.add_argument(
        "--history-file",
        help="File history della shell da cui recuperare l'ultimo comando se non passi input esplicito",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modello Groq da usare. Default economico e veloce: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Numero massimo di iterazioni tool/model. Default: {DEFAULT_MAX_STEPS}",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()

    if args.from_file:
        capture = read_capture_file(
            file_path=args.from_file,
            cwd=workspace,
            command=args.captured_command,
            exit_code=args.captured_exit_code,
        )
        save_capture(workspace, capture)
    elif args.command:
        capture = run_command(args.command, cwd=workspace, use_shell=args.shell)
        save_capture(workspace, capture)
    elif not sys.stdin.isatty():
        capture = read_stdin_capture(cwd=workspace)
        save_capture(workspace, capture)
    elif args.log_file:
        capture = find_recent_log_capture(workspace, explicit_log_file=args.log_file)
        if capture is None:
            parser.error("Nessun errore rilevato nel file di log specificato")
    else:
        capture = replay_last_shell_command(workspace, explicit_history_file=args.history_file)
        if capture is not None:
            save_capture(workspace, capture)
        if capture is None:
            capture = find_recent_log_capture(workspace)
        if capture is None:
            capture = load_saved_capture(workspace)
        if capture is None:
            parser.error(
                "Passa --command, invia l'errore via stdin, usa --log-file oppure assicurati che l'ultimo comando sia presente nella history della shell"
            )

    agent = DebugAgent(workspace=workspace, model=args.model)
    print(
        agent.analyze(
            capture=capture,
            max_steps=args.max_steps,
            allow_workspace_context=args.allow_workspace_context,
            print_prompt=args.print_prompt,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())