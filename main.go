package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	systemPrompt = `You are a CLI debugging agent powered by Groq.

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
- do not repeat the error text verbatim unless necessary`
	defaultModel               = "llama-3.3-70b-versatile"
	defaultMaxSteps            = 6
	defaultOutputLimit         = 12000
	defaultMaxCompletionTokens = 350
	cacheDirName               = ".cli_debug_agent"
	lastCaptureFilename        = ".last_debug_capture.json"
	redactedPath               = "<PATH>"
	redactedHome               = "<HOME>"
	redactedWorkspace          = "<WORKSPACE>"
	redactedAPIKey             = "<API_KEY>"
	groqEndpoint               = "https://api.groq.com/openai/v1/chat/completions"
)

var (
	bashHistoryCandidates = []string{".bash_history", ".zhistory"}
	errorPattern          = regexp.MustCompile(`(?i)(exception|error|fatal|failed|panic|cannot |can't |no such file|not found|permission denied|module not found|command not found)`)
	sensitivePatterns     = []struct {
		re          *regexp.Regexp
		replacement string
	}{
		{re: regexp.MustCompile(`gsk_[A-Za-z0-9_\-]+`), replacement: redactedAPIKey},
		{re: regexp.MustCompile(`\bsk-[A-Za-z0-9_\-]+\b`), replacement: redactedAPIKey},
		{re: regexp.MustCompile(`\bAKIA[0-9A-Z]{16}\b`), replacement: redactedAPIKey},
		{re: regexp.MustCompile(`(?m)(^|[^A-Za-z0-9_./-])(/(?:[^\s/:]+/)*[^\s:]+)`), replacement: `${1}` + redactedPath},
	}
)

type commandCapture struct {
	Command  string `json:"command"`
	Cwd      string `json:"cwd"`
	ExitCode int    `json:"exit_code"`
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	Source   string `json:"source"`
}

type workspaceTools struct {
	workspace string
}

type toolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function toolCallFunction `json:"function"`
}

type responseMessage struct {
	Content   string     `json:"content"`
	ToolCalls []toolCall `json:"tool_calls"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message responseMessage `json:"message"`
	} `json:"choices"`
}

type groqClient struct {
	apiKey     string
	httpClient *http.Client
}

type debugAgent struct {
	client    *groqClient
	model     string
	workspace string
	tools     *workspaceTools
	toolSpecs []map[string]any
	toolMap   map[string]func(map[string]any) (string, error)
}

type cliOptions struct {
	Command               string
	UseShell              bool
	Workspace             string
	LogFile               string
	FromFile              string
	CapturedCommand       string
	CapturedExitCode      int
	AllowWorkspaceContext bool
	PrintPrompt           bool
	HistoryFile           string
	Model                 string
	MaxSteps              int
}

type fileCandidate struct {
	path    string
	modTime time.Time
}

func safePath(workspace string, relativePath string) (string, error) {
	workspaceRoot, err := filepath.Abs(workspace)
	if err != nil {
		return "", err
	}

	candidate, err := filepath.Abs(filepath.Join(workspaceRoot, relativePath))
	if err != nil {
		return "", err
	}

	if candidate == workspaceRoot {
		return candidate, nil
	}

	prefix := workspaceRoot + string(os.PathSeparator)
	if !strings.HasPrefix(candidate, prefix) {
		return "", errors.New("path fuori dalla workspace")
	}

	return candidate, nil
}

func truncate(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	omitted := len(text) - limit
	return fmt.Sprintf("%s\n\n...[troncato, %d caratteri omessi]", text[:limit], omitted)
}

func readTextTail(path string, limit int64) (string, error) {
	handle, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer handle.Close()

	info, err := handle.Stat()
	if err != nil {
		return "", err
	}

	start := info.Size() - limit
	if start < 0 {
		start = 0
	}

	if _, err := handle.Seek(start, io.SeekStart); err != nil {
		return "", err
	}

	data, err := io.ReadAll(handle)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

func extractLatestErrorBlock(text string) string {
	const tracebackMarker = "Traceback (most recent call last):"
	if idx := strings.LastIndex(text, tracebackMarker); idx >= 0 {
		return strings.TrimSpace(text[idx:])
	}

	lines := strings.Split(text, "\n")
	for idx := len(lines) - 1; idx >= 0; idx-- {
		if errorPattern.MatchString(lines[idx]) {
			start := idx - 20
			if start < 0 {
				start = 0
			}
			end := idx + 8
			if end > len(lines) {
				end = len(lines)
			}
			return strings.TrimSpace(strings.Join(lines[start:end], "\n"))
		}
	}

	return ""
}

func sanitizeText(text string, workspace string, cwd string) string {
	sanitized := text
	replacements := [][2]string{}

	if home, err := os.UserHomeDir(); err == nil && home != "" {
		replacements = append(replacements, [2]string{home, redactedHome})
	}
	if workspace != "" {
		replacements = append(replacements, [2]string{workspace, redactedWorkspace})
	}
	if cwd != "" {
		replacements = append(replacements, [2]string{cwd, "<CWD>"})
	}

	sort.Slice(replacements, func(i int, j int) bool {
		return len(replacements[i][0]) > len(replacements[j][0])
	})

	for _, pair := range replacements {
		if pair[0] == "" {
			continue
		}
		sanitized = strings.ReplaceAll(sanitized, pair[0], pair[1])
	}

	for _, pattern := range sensitivePatterns {
		sanitized = pattern.re.ReplaceAllString(sanitized, pattern.replacement)
	}

	return sanitized
}

func resolveAPIKey() (string, error) {
	apiKey := strings.TrimSpace(os.Getenv("GROQ_API_KEY"))
	if apiKey != "" {
		return apiKey, nil
	}
	return "", errors.New("imposta GROQ_API_KEY")
}

func (capture commandCapture) modelPayload(workspace string, allowWorkspaceContext bool) string {
	errorText := strings.TrimSpace(capture.Stderr)
	if errorText == "" {
		errorText = strings.TrimSpace(capture.Stdout)
	}
	errorBlock := extractLatestErrorBlock(errorText)
	if errorBlock == "" {
		errorBlock = errorText
	}
	if errorBlock == "" {
		errorBlock = "Nessun output di errore disponibile"
	}

	payload := map[string]any{
		"source":    capture.Source,
		"exit_code": capture.ExitCode,
		"command":   sanitizeText(capture.Command, workspace, capture.Cwd),
		"error":     sanitizeText(truncate(errorBlock, defaultOutputLimit), workspace, capture.Cwd),
	}
	if allowWorkspaceContext {
		payload["workspace_context"] = "enabled"
	}

	data, _ := json.MarshalIndent(payload, "", "  ")
	return string(data)
}

func (capture commandCapture) toJSON() string {
	data, _ := json.MarshalIndent(capture, "", "  ")
	return string(data)
}

func commandCaptureFromJSON(content string) (commandCapture, error) {
	var capture commandCapture
	err := json.Unmarshal([]byte(content), &capture)
	if capture.Source == "" {
		capture.Source = "command"
	}
	return capture, err
}

func getLastCapturePath(workspace string) string {
	return filepath.Join(workspace, cacheDirName, lastCaptureFilename)
}

func getLegacyCapturePath(workspace string) string {
	return filepath.Join(workspace, "cli_debug_agent", lastCaptureFilename)
}

func saveCapture(workspace string, capture commandCapture) error {
	cachePath := getLastCapturePath(workspace)
	if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
		return err
	}
	return os.WriteFile(cachePath, []byte(capture.toJSON()), 0o644)
}

func loadSavedCapture(workspace string) (*commandCapture, error) {
	cachePaths := []string{getLastCapturePath(workspace), getLegacyCapturePath(workspace)}
	var content []byte
	var err error
	for _, cachePath := range cachePaths {
		content, err = os.ReadFile(cachePath)
		if err == nil {
			capture, err := commandCaptureFromJSON(string(content))
			if err != nil {
				return nil, err
			}
			return &capture, nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
	}

	return nil, nil
}

func resolveHistoryFile(explicit string) string {
	if explicit != "" {
		path := filepath.Clean(expandUser(explicit))
		if fileExists(path) {
			return path
		}
		return ""
	}

	if envHistory := strings.TrimSpace(os.Getenv("HISTFILE")); envHistory != "" {
		path := filepath.Clean(expandUser(envHistory))
		if fileExists(path) {
			return path
		}
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}

	for _, candidate := range bashHistoryCandidates {
		path := filepath.Join(home, candidate)
		if fileExists(path) {
			return path
		}
	}

	return ""
}

func isDebugAgentCommand(command string) bool {
	lowered := strings.ToLower(command)
	return strings.Contains(lowered, "debug_agent.py") ||
		strings.Contains(lowered, "cli_debug_agent") ||
		strings.Contains(lowered, "cli-debug-agent") ||
		strings.HasPrefix(lowered, "dbg ") ||
		lowered == "dbg"
}

func findLastShellCommand(explicitHistoryFile string) string {
	historyFile := resolveHistoryFile(explicitHistoryFile)
	if historyFile == "" {
		return ""
	}

	content, err := os.ReadFile(historyFile)
	if err != nil {
		return ""
	}

	lines := strings.Split(string(content), "\n")
	for idx := len(lines) - 1; idx >= 0; idx-- {
		command := strings.TrimSpace(lines[idx])
		if command == "" {
			continue
		}
		if strings.HasPrefix(command, ": ") && strings.Contains(command, ";") {
			parts := strings.SplitN(command, ";", 2)
			if len(parts) == 2 {
				command = strings.TrimSpace(parts[1])
			}
		}
		if command == "" || isDebugAgentCommand(command) {
			continue
		}
		return command
	}

	return ""
}

func replayLastShellCommand(cwd string, explicitHistoryFile string) (*commandCapture, error) {
	command := findLastShellCommand(explicitHistoryFile)
	if command == "" {
		return nil, nil
	}

	capture, err := runCommand(command, cwd, true)
	if err != nil {
		return nil, err
	}
	capture.Source = "history_replay"
	return &capture, nil
}

func findRecentLogCapture(workspace string, explicitLogFile string) (*commandCapture, error) {
	var candidateFiles []fileCandidate

	if explicitLogFile != "" {
		logPath, err := safePath(workspace, explicitLogFile)
		if err != nil {
			return nil, err
		}
		if info, err := os.Stat(logPath); err == nil && !info.IsDir() {
			candidateFiles = append(candidateFiles, fileCandidate{path: logPath, modTime: info.ModTime()})
		}
	} else {
		err := filepath.WalkDir(workspace, func(path string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if entry.IsDir() {
				if entry.Name() == ".git" || entry.Name() == "bin" || entry.Name() == "__pycache__" {
					return filepath.SkipDir
				}
				return nil
			}

			name := entry.Name()
			if name != "nohup.out" && !strings.HasSuffix(name, ".log") && !strings.HasSuffix(name, ".err") && !strings.HasSuffix(name, ".out") {
				return nil
			}

			info, err := entry.Info()
			if err != nil {
				return nil
			}
			candidateFiles = append(candidateFiles, fileCandidate{path: path, modTime: info.ModTime()})
			return nil
		})
		if err != nil {
			return nil, err
		}

		sort.Slice(candidateFiles, func(i int, j int) bool {
			if candidateFiles[i].modTime.Equal(candidateFiles[j].modTime) {
				return candidateFiles[i].path < candidateFiles[j].path
			}
			return candidateFiles[i].modTime.After(candidateFiles[j].modTime)
		})
		if len(candidateFiles) > 30 {
			candidateFiles = candidateFiles[:30]
		}
	}

	for _, candidate := range candidateFiles {
		tail, err := readTextTail(candidate.path, defaultOutputLimit)
		if err != nil {
			continue
		}

		errorBlock := extractLatestErrorBlock(tail)
		if errorBlock == "" {
			continue
		}

		relPath, err := filepath.Rel(workspace, candidate.path)
		if err != nil {
			relPath = filepath.Base(candidate.path)
		}

		capture := commandCapture{
			Source:   "log_file",
			Command:  "log:" + filepath.ToSlash(relPath),
			Cwd:      workspace,
			ExitCode: 1,
			Stdout:   "",
			Stderr:   errorBlock,
		}
		return &capture, nil
	}

	return nil, nil
}

func newWorkspaceTools(workspace string) *workspaceTools {
	return &workspaceTools{workspace: workspace}
}

func (tools *workspaceTools) listFiles(path string) (string, error) {
	if path == "" {
		path = "."
	}
	target, err := safePath(tools.workspace, path)
	if err != nil {
		return "", err
	}

	info, err := os.Stat(target)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", errors.New("il percorso non e' una directory")
	}

	entries, err := os.ReadDir(target)
	if err != nil {
		return "", err
	}

	result := make([]string, 0, len(entries))
	for _, child := range entries {
		relativeName, err := filepath.Rel(tools.workspace, filepath.Join(target, child.Name()))
		if err != nil {
			continue
		}
		name := filepath.ToSlash(relativeName)
		if child.IsDir() {
			name += "/"
		}
		result = append(result, name)
	}

	payload, _ := json.Marshal(map[string]any{"path": path, "entries": result})
	return string(payload), nil
}

func (tools *workspaceTools) readFile(path string, startLine int, endLine int) (string, error) {
	target, err := safePath(tools.workspace, path)
	if err != nil {
		return "", err
	}

	info, err := os.Stat(target)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", errors.New("il percorso non e' un file")
	}

	content, err := os.ReadFile(target)
	if err != nil {
		return "", err
	}

	lines := strings.SplitAfter(string(content), "\n")
	if startLine < 1 {
		startLine = 1
	}
	if endLine < startLine {
		endLine = startLine
	}
	if startLine > len(lines) {
		startLine = len(lines)
	}
	if endLine > len(lines) {
		endLine = len(lines)
	}

	selected := ""
	if len(lines) > 0 && startLine > 0 {
		selected = strings.Join(lines[startLine-1:endLine], "")
	}

	payload, _ := json.Marshal(map[string]any{
		"path":       path,
		"start_line": startLine,
		"end_line":   endLine,
		"content":    selected,
	})
	return string(payload), nil
}

func (tools *workspaceTools) searchText(query string, path string, maxResults int) (string, error) {
	if path == "" {
		path = "."
	}
	if maxResults <= 0 {
		maxResults = 20
	}

	target, err := safePath(tools.workspace, path)
	if err != nil {
		return "", err
	}

	matches := make([]map[string]any, 0, maxResults)
	lowered := strings.ToLower(query)

	appendMatches := func(filePath string) error {
		file, err := os.Open(filePath)
		if err != nil {
			return nil
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)
		buffer := make([]byte, 0, 64*1024)
		scanner.Buffer(buffer, 1024*1024)
		lineNumber := 0
		for scanner.Scan() {
			lineNumber++
			line := scanner.Text()
			if strings.Contains(strings.ToLower(line), lowered) {
				relativePath, err := filepath.Rel(tools.workspace, filePath)
				if err != nil {
					relativePath = filePath
				}
				matches = append(matches, map[string]any{
					"path": filepath.ToSlash(relativePath),
					"line": lineNumber,
					"text": line,
				})
				if len(matches) >= maxResults {
					return io.EOF
				}
			}
		}
		return nil
	}

	info, err := os.Stat(target)
	if err != nil {
		return "", err
	}

	if !info.IsDir() {
		err = appendMatches(target)
	} else {
		err = filepath.WalkDir(target, func(filePath string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if entry.IsDir() {
				return nil
			}
			return appendMatches(filePath)
		})
	}
	if err != nil && !errors.Is(err, io.EOF) {
		return "", err
	}

	payload, _ := json.Marshal(map[string]any{"query": query, "matches": matches})
	return string(payload), nil
}

func newGroqClient(apiKey string) *groqClient {
	return &groqClient{
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (client *groqClient) createChatCompletion(requestBody map[string]any) (*chatCompletionResponse, error) {
	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(http.MethodPost, groqEndpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+client.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("richiesta Groq fallita (%s): %s", resp.Status, strings.TrimSpace(string(responseBody)))
	}

	var payload chatCompletionResponse
	if err := json.Unmarshal(responseBody, &payload); err != nil {
		return nil, err
	}
	if len(payload.Choices) == 0 {
		return nil, errors.New("Groq non ha restituito scelte")
	}
	return &payload, nil
}

func newDebugAgent(workspace string, model string) (*debugAgent, error) {
	apiKey, err := resolveAPIKey()
	if err != nil {
		return nil, err
	}

	tools := newWorkspaceTools(workspace)
	agent := &debugAgent{
		client:    newGroqClient(apiKey),
		model:     model,
		workspace: workspace,
		tools:     tools,
		toolSpecs: []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "list_files",
					"description": "Elenca file e directory nella workspace o in una sottodirectory.",
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"path": map[string]any{"type": "string", "description": "Percorso relativo alla workspace", "default": "."},
						},
					},
				},
			},
			{
				"type": "function",
				"function": map[string]any{
					"name":        "read_file",
					"description": "Legge righe di un file della workspace per verificare stacktrace, import o configurazioni.",
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"path":       map[string]any{"type": "string", "description": "Percorso relativo del file"},
							"start_line": map[string]any{"type": "integer", "default": 1},
							"end_line":   map[string]any{"type": "integer", "default": 250},
						},
						"required": []string{"path"},
					},
				},
			},
			{
				"type": "function",
				"function": map[string]any{
					"name":        "search_text",
					"description": "Cerca testo nella workspace per trovare riferimenti a errori, simboli, import o configurazioni.",
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"query":       map[string]any{"type": "string", "description": "Testo da cercare"},
							"path":        map[string]any{"type": "string", "default": "."},
							"max_results": map[string]any{"type": "integer", "default": 20},
						},
						"required": []string{"query"},
					},
				},
			},
		},
	}

	agent.toolMap = map[string]func(map[string]any) (string, error){
		"list_files": func(arguments map[string]any) (string, error) {
			return agent.tools.listFiles(stringArg(arguments, "path", "."))
		},
		"read_file": func(arguments map[string]any) (string, error) {
			return agent.tools.readFile(
				stringArg(arguments, "path", ""),
				intArg(arguments, "start_line", 1),
				intArg(arguments, "end_line", 250),
			)
		},
		"search_text": func(arguments map[string]any) (string, error) {
			return agent.tools.searchText(
				stringArg(arguments, "query", ""),
				stringArg(arguments, "path", "."),
				intArg(arguments, "max_results", 20),
			)
		},
	}

	return agent, nil
}

func (agent *debugAgent) analyze(capture commandCapture, maxSteps int, allowWorkspaceContext bool, printPrompt bool) (string, error) {
	var userPrompt string
	if allowWorkspaceContext {
		userPrompt = "Analizza questo errore di terminale. I dati sensibili sono gia' stati sanitizzati. Se serve, usa i tool per ispezionare i file della workspace e confermare la diagnosi.\n\nDettagli esecuzione:\n" + capture.modelPayload(agent.workspace, true)
	} else {
		userPrompt = "Analizza questo errore di terminale usando solo il payload sanitizzato seguente. Non hai accesso alla workspace e devi evitare assunzioni non supportate dai dati.\n\nDettagli esecuzione:\n" + capture.modelPayload(agent.workspace, false)
	}

	messages := []map[string]any{
		{"role": "system", "content": systemPrompt},
		{"role": "user", "content": userPrompt},
	}

	for step := 0; step < maxSteps; step++ {
		request := map[string]any{
			"model":                 agent.model,
			"messages":              messages,
			"temperature":           0.1,
			"max_completion_tokens": defaultMaxCompletionTokens,
		}
		if allowWorkspaceContext {
			request["tools"] = agent.toolSpecs
			request["tool_choice"] = "auto"
		}

		if printPrompt {
			preview, _ := json.MarshalIndent(request, "", "  ")
			fmt.Fprintln(os.Stderr, "=== MODEL REQUEST BEGIN ===")
			fmt.Fprintln(os.Stderr, string(preview))
			fmt.Fprintln(os.Stderr, "=== MODEL REQUEST END ===")
		}

		completion, err := agent.client.createChatCompletion(request)
		if err != nil {
			return "", err
		}

		message := completion.Choices[0].Message
		assistantPayload := map[string]any{"role": "assistant", "content": message.Content}
		if len(message.ToolCalls) > 0 {
			assistantPayload["tool_calls"] = message.ToolCalls
		}
		messages = append(messages, assistantPayload)

		if len(message.ToolCalls) == 0 {
			if strings.TrimSpace(message.Content) == "" {
				return "Nessuna diagnosi generata", nil
			}
			return message.Content, nil
		}

		for _, call := range message.ToolCalls {
			arguments := map[string]any{}
			if strings.TrimSpace(call.Function.Arguments) != "" {
				if err := json.Unmarshal([]byte(call.Function.Arguments), &arguments); err != nil {
					arguments = nil
				}
			}

			content := ""
			if arguments == nil {
				errorPayload, _ := json.Marshal(map[string]any{"error": "Argomenti JSON non validi"})
				content = string(errorPayload)
			} else {
				handler := agent.toolMap[call.Function.Name]
				if handler == nil {
					errorPayload, _ := json.Marshal(map[string]any{"error": "Tool non supportato: " + call.Function.Name})
					content = string(errorPayload)
				} else {
					toolResult, err := handler(arguments)
					if err != nil {
						errorPayload, _ := json.Marshal(map[string]any{"error": fmt.Sprintf("Errore durante %s: %v", call.Function.Name, err)})
						content = string(errorPayload)
					} else {
						content = toolResult
					}
				}
			}

			messages = append(messages, map[string]any{
				"role":         "tool",
				"tool_call_id": call.ID,
				"name":         call.Function.Name,
				"content":      sanitizeText(content, agent.workspace, ""),
			})
		}
	}

	return "", fmt.Errorf("numero massimo di step raggiunto (%d)", maxSteps)
}

func runCommand(command string, cwd string, useShell bool) (commandCapture, error) {
	var cmd *exec.Cmd
	if useShell {
		cmd = exec.Command("bash", "-lc", command)
	} else {
		parts, err := shellSplit(command)
		if err != nil {
			return commandCapture{}, err
		}
		if len(parts) == 0 {
			return commandCapture{}, errors.New("comando vuoto")
		}
		cmd = exec.Command(parts[0], parts[1:]...)
	}

	cmd.Dir = cwd
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	exitCode := 0
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			return commandCapture{}, err
		}
	}

	return commandCapture{
		Command:  command,
		Cwd:      cwd,
		ExitCode: exitCode,
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Source:   "command",
	}, nil
}

func readStdinCapture(cwd string) (commandCapture, error) {
	content, err := io.ReadAll(os.Stdin)
	if err != nil {
		return commandCapture{}, err
	}
	if strings.TrimSpace(string(content)) == "" {
		return commandCapture{}, errors.New("nessun input ricevuto da stdin")
	}
	return commandCapture{
		Command:  "stdin",
		Cwd:      cwd,
		ExitCode: 1,
		Stdout:   "",
		Stderr:   string(content),
		Source:   "stdin",
	}, nil
}

func readCaptureFile(filePath string, cwd string, command string, exitCode int) (commandCapture, error) {
	path := filepath.Clean(expandUser(filePath))
	content, err := os.ReadFile(path)
	if err != nil {
		return commandCapture{}, err
	}
	if strings.TrimSpace(string(content)) == "" {
		return commandCapture{}, errors.New("il file di cattura e' vuoto")
	}

	resolvedCommand := command
	if resolvedCommand == "" {
		resolvedCommand = "capture:" + path
	}

	return commandCapture{
		Command:  resolvedCommand,
		Cwd:      cwd,
		ExitCode: exitCode,
		Stdout:   "",
		Stderr:   string(content),
		Source:   "capture_file",
	}, nil
}

func parseArgs(args []string) (cliOptions, error) {
	options := cliOptions{}
	fs := flag.NewFlagSet("cli-debug-agent", flag.ContinueOnError)
	var stderr bytes.Buffer
	fs.SetOutput(&stderr)

	fs.StringVar(&options.Command, "command", "", "Comando da eseguire e analizzare. Esempio: --command 'python app.py'")
	fs.BoolVar(&options.UseShell, "shell", false, "Esegue il comando tramite shell. Utile per pipe, redirect e comandi composti.")
	fs.StringVar(&options.Workspace, "workspace", ".", "Workspace visibile all'agente. Default: directory corrente")
	fs.StringVar(&options.LogFile, "log-file", "", "File di log da cui leggere automaticamente l'ultimo traceback o errore")
	fs.StringVar(&options.FromFile, "from-file", "", "File contenente output di terminale gia' catturato da analizzare")
	fs.StringVar(&options.CapturedCommand, "captured-command", "", "Comando originale associato al contenuto passato con --from-file")
	fs.IntVar(&options.CapturedExitCode, "captured-exit-code", 1, "Exit code del comando associato a --from-file. Default: 1")
	fs.BoolVar(&options.AllowWorkspaceContext, "allow-workspace-context", false, "Permette al modello di usare tool locali della workspace.")
	fs.BoolVar(&options.PrintPrompt, "print-prompt", false, "Stampa su stderr la request sanitizzata inviata al modello.")
	fs.StringVar(&options.HistoryFile, "history-file", "", "File history della shell da cui recuperare l'ultimo comando")
	fs.StringVar(&options.Model, "model", defaultModel, "Modello Groq da usare")
	fs.IntVar(&options.MaxSteps, "max-steps", defaultMaxSteps, "Numero massimo di iterazioni tool/model")

	if err := fs.Parse(args); err != nil {
		return options, errors.New(strings.TrimSpace(stderr.String()))
	}
	if fs.NArg() != 0 {
		return options, fmt.Errorf("argomenti posizionali non supportati: %s", strings.Join(fs.Args(), " "))
	}

	return options, nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	options, err := parseArgs(os.Args[1:])
	if err != nil {
		return err
	}

	workspace, err := filepath.Abs(options.Workspace)
	if err != nil {
		return err
	}

	var capture commandCapture
	switch {
	case options.FromFile != "":
		capture, err = readCaptureFile(options.FromFile, workspace, options.CapturedCommand, options.CapturedExitCode)
		if err != nil {
			return err
		}
		if err := saveCapture(workspace, capture); err != nil {
			return err
		}
	case options.Command != "":
		capture, err = runCommand(options.Command, workspace, options.UseShell)
		if err != nil {
			return err
		}
		if err := saveCapture(workspace, capture); err != nil {
			return err
		}
	case !isTerminal(os.Stdin):
		capture, err = readStdinCapture(workspace)
		if err != nil {
			return err
		}
		if err := saveCapture(workspace, capture); err != nil {
			return err
		}
	case options.LogFile != "":
		loaded, err := findRecentLogCapture(workspace, options.LogFile)
		if err != nil {
			return err
		}
		if loaded == nil {
			return errors.New("nessun errore rilevato nel file di log specificato")
		}
		capture = *loaded
	default:
		loaded, err := replayLastShellCommand(workspace, options.HistoryFile)
		if err != nil {
			return err
		}
		if loaded != nil {
			capture = *loaded
			if err := saveCapture(workspace, capture); err != nil {
				return err
			}
		} else {
			loaded, err = findRecentLogCapture(workspace, "")
			if err != nil {
				return err
			}
			if loaded != nil {
				capture = *loaded
			} else {
				saved, err := loadSavedCapture(workspace)
				if err != nil {
					return err
				}
				if saved == nil {
					return errors.New("passa --command, invia l'errore via stdin, usa --log-file oppure assicurati che l'ultimo comando sia presente nella history della shell")
				}
				capture = *saved
			}
		}
	}

	agent, err := newDebugAgent(workspace, options.Model)
	if err != nil {
		return err
	}

	result, err := agent.analyze(capture, options.MaxSteps, options.AllowWorkspaceContext, options.PrintPrompt)
	if err != nil {
		return err
	}

	fmt.Println(result)
	return nil
}

func shellSplit(command string) ([]string, error) {
	var parts []string
	var current strings.Builder
	inSingle := false
	inDouble := false
	escaped := false

	flush := func() {
		if current.Len() > 0 {
			parts = append(parts, current.String())
			current.Reset()
		}
	}

	for _, ch := range command {
		switch {
		case escaped:
			current.WriteRune(ch)
			escaped = false
		case ch == '\\' && !inSingle:
			escaped = true
		case ch == '\'' && !inDouble:
			inSingle = !inSingle
		case ch == '"' && !inSingle:
			inDouble = !inDouble
		case (ch == ' ' || ch == '\t' || ch == '\n') && !inSingle && !inDouble:
			flush()
		default:
			current.WriteRune(ch)
		}
	}

	if escaped || inSingle || inDouble {
		return nil, errors.New("impossibile analizzare il comando: quoting non bilanciato")
	}
	flush()
	return parts, nil
}

func expandUser(path string) string {
	if path == "~" || strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			if path == "~" {
				return home
			}
			return filepath.Join(home, strings.TrimPrefix(path, "~/"))
		}
	}
	return path
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func stringArg(arguments map[string]any, key string, defaultValue string) string {
	value, ok := arguments[key]
	if !ok {
		return defaultValue
	}
	stringValue, ok := value.(string)
	if !ok || stringValue == "" {
		return defaultValue
	}
	return stringValue
}

func intArg(arguments map[string]any, key string, defaultValue int) int {
	value, ok := arguments[key]
	if !ok {
		return defaultValue
	}
	switch typed := value.(type) {
	case float64:
		return int(typed)
	case int:
		return typed
	case string:
		parsed, err := strconv.Atoi(typed)
		if err == nil {
			return parsed
		}
	}
	return defaultValue
}

func isTerminal(file *os.File) bool {
	info, err := file.Stat()
	if err != nil {
		return false
	}
	return (info.Mode() & os.ModeCharDevice) != 0
}
