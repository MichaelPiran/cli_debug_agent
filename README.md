# CLI Debug Agent

Piccolo agente CLI scritto in Go che cattura l'output di un comando fallito e lo fa analizzare da Groq.

## Requisiti

- Go 1.22+
- una chiave Groq in `GROQ_API_KEY`

## Build

```bash
make build
```

L'eseguibile viene generato in `bin/cli-debug-agent`.

## Uso rapido

```bash
source cli_debug_agent/dbg.sh
make build
dbg ls /percorso/inesistente
```

## Uso globale

```bash
echo 'source /home/piran/code/mp_git/cli_debug_agent/dbg.sh' >> ~/.bashrc
source ~/.bashrc
```

Per vedere il prompt sanitizzato inviato al modello:

```bash
dbg --print-prompt ls /percorso/inesistente
```

Con comandi shell composti:

```bash
dbg -- bash -lc 'kubectl get pod -n pippo | grep api'
```

## Uso diretto del binario

```bash
bin/cli-debug-agent --command 'ls /percorso/inesistente' --workspace .
```

## Note

- Di default al modello viene inviato solo un errore sanitizzato.
- Il modello predefinito e' `llama-3.3-70b-versatile`.
- Per dare accesso anche alla workspace: `dbg --allow-workspace-context ...`
