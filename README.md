# CLI Debug Agent

Piccolo agente CLI che cattura l'output di un comando fallito e lo fa analizzare da Groq.

## Requisiti

```bash
pip install groq
```

Serve anche una chiave Groq in `GROQ_API_KEY` oppure una funzione `get_groq_api_key()` in `utis.py`.

## Uso rapido

```bash
source cli_debug_agent/dbg.sh
dbg python3 app.py
```

Per vedere il prompt sanitizzato inviato al modello:

```bash
dbg --print-prompt python3 app.py
```

Con comandi shell composti:

```bash
dbg -- bash -lc 'kubectl get pod -n pippo | grep api'
```

## Note

- Di default al modello viene inviato solo un errore sanitizzato.
- Il modello predefinito e' economico e veloce: `llama-3.1-8b-instant`.
- Per dare accesso anche alla workspace: `dbg --allow-workspace-context ...`
