/**
 * AgentCity tools for pi.
 *
 * A) Bridge tools that delegate to the Python runner:
 *    `<python> -m agents.tools.runner <tool_name>` with the tool arguments
 *    serialized as JSON on stdin. The runner prints a single JSON line:
 *    `{"text": "..."}` on success or `{"error": "..."}` on failure.
 *
 * B) A `task` tool for sub-agent delegation: spawns a child pi process in
 *    print mode with this same extension loaded, using subagent definitions
 *    from the JSON file at AGENTCITY_SUBAGENTS_FILE.
 */

import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const DEFAULT_TIMEOUT_MS = 120_000;
const TASK_DEFAULT_TIMEOUT_MS = 1_800_000;

const extensionPath = fileURLToPath(import.meta.url);
const extensionDir = dirname(extensionPath);

function agentCityCwd(): string {
	return process.env.AGENTCITY_ROOT || process.cwd();
}

interface ProcessResult {
	stdout: string;
	stderr: string;
	code: number | null;
	timedOut: boolean;
}

function runProcess(
	command: string,
	args: string[],
	options: { cwd: string; timeoutMs: number; input?: string },
): Promise<ProcessResult> {
	return new Promise((resolvePromise, rejectPromise) => {
		const child = spawn(command, args, { cwd: options.cwd });

		let stdout = "";
		let stderr = "";
		let timedOut = false;
		let settled = false;

		const timer = setTimeout(() => {
			timedOut = true;
			child.kill("SIGTERM");
			// Give the child a moment to exit, then force-kill.
			setTimeout(() => {
				if (!settled) child.kill("SIGKILL");
			}, 5000).unref();
		}, options.timeoutMs);

		child.stdout.on("data", (chunk) => {
			stdout += chunk.toString();
		});
		child.stderr.on("data", (chunk) => {
			stderr += chunk.toString();
		});
		child.on("error", (err) => {
			settled = true;
			clearTimeout(timer);
			rejectPromise(err);
		});
		child.on("close", (code) => {
			settled = true;
			clearTimeout(timer);
			resolvePromise({ stdout, stderr, code, timedOut });
		});

		if (options.input !== undefined) {
			child.stdin.write(options.input);
		}
		child.stdin.end();
	});
}

function stderrExcerpt(stderr: string, maxChars = 2000): string {
	const trimmed = stderr.trim();
	if (!trimmed) return "(no stderr output)";
	return trimmed.length > maxChars ? `...${trimmed.slice(-maxChars)}` : trimmed;
}

/** Run one bridge tool through the Python runner and return its text. */
async function runBridgeTool(toolName: string, params: unknown, timeoutMs: number): Promise<string> {
	const python = process.env.AGENTCITY_PYTHON || "python";
	const result = await runProcess(python, ["-m", "agents.tools.runner", toolName], {
		cwd: agentCityCwd(),
		timeoutMs,
		input: JSON.stringify(params ?? {}),
	});

	if (result.timedOut) {
		throw new Error(
			`${toolName} timed out after ${Math.round(timeoutMs / 1000)}s. stderr: ${stderrExcerpt(result.stderr)}`,
		);
	}
	if (result.code !== 0) {
		throw new Error(`${toolName} exited with code ${result.code}. stderr: ${stderrExcerpt(result.stderr)}`);
	}

	const output = result.stdout.trim();
	let parsed: { text?: string; error?: string };
	try {
		parsed = JSON.parse(output);
	} catch {
		throw new Error(
			`${toolName} produced unparseable output: ${output.slice(0, 2000) || "(empty)"}. stderr: ${stderrExcerpt(result.stderr)}`,
		);
	}

	if (typeof parsed.error === "string" && parsed.error) {
		throw new Error(parsed.error);
	}
	if (typeof parsed.text === "string") {
		return parsed.text;
	}
	throw new Error(`${toolName} returned unexpected JSON: ${output.slice(0, 2000)}`);
}

interface BridgeToolDef {
	name: string;
	label: string;
	description: string;
	parameters: Parameters<ExtensionAPI["registerTool"]>[0]["parameters"];
	timeoutMs?: number;
}

const bridgeTools: BridgeToolDef[] = [
	{
		name: "catalog_article",
		label: "Catalog Article",
		description: "Persist metadata about a located research paper and its saved PDF",
		parameters: Type.Object({
			title: Type.String({ description: "Paper title" }),
			conference: Type.Optional(Type.String({ description: "Conference or venue" })),
			datasets: Type.Optional(Type.String({ description: "Datasets used in the paper" })),
			repo_url: Type.Optional(Type.String({ description: "URL of the paper's code repository" })),
			pdf_path: Type.Optional(Type.String({ description: "Local path to the saved PDF" })),
			notes: Type.Optional(Type.String({ description: "Additional notes" })),
			model_name: Type.Optional(Type.String({ description: "Name of the model proposed in the paper" })),
		}),
	},
	{
		name: "search_paper",
		label: "Search Paper",
		description: "Search for academic papers on arXiv. Returns title, abstract, arxiv_id, URLs.",
		parameters: Type.Object({
			query: Type.String({ description: "Search query" }),
			num: Type.Optional(Type.Number({ description: "Number of results to return" })),
			end_date: Type.Optional(Type.String({ description: "Only include papers published before this date" })),
		}),
	},
	{
		name: "evaluate_paper_relevance",
		label: "Evaluate Paper Relevance",
		description:
			"Record a paper relevance evaluation request; use your own judgment to score relevance",
		parameters: Type.Object({
			title: Type.String({ description: "Paper title" }),
			abstract: Type.String({ description: "Paper abstract" }),
			user_query: Type.String({ description: "The user's research query to evaluate relevance against" }),
		}),
	},
	{
		name: "test_migration",
		label: "Test Migration",
		description:
			"Run the LibCity benchmark with the provided model and dataset (supports GPU acceleration)",
		parameters: Type.Object({
			model_name: Type.String({ description: "Name of the migrated model to test" }),
			dataset: Type.Optional(Type.String({ description: "Dataset to run the benchmark on" })),
			task: Type.Optional(Type.String({ description: "LibCity task name" })),
			paper_title: Type.Optional(Type.String({ description: "Title of the source paper" })),
			gpu: Type.Optional(Type.String({ description: "GPU device to use, e.g. '0' or 'cpu'" })),
		}),
		timeoutMs: 3_600_000,
	},
	{
		name: "tune_migration_model",
		label: "Tune Migration Model",
		description: "Hyperparameter tuning for migrated LibCity models",
		parameters: Type.Object({
			model_name: Type.String({ description: "Name of the migrated model to tune" }),
			dataset: Type.Optional(Type.String({ description: "Dataset to tune on" })),
			config_file: Type.Optional(Type.String({ description: "Path to a config file" })),
			task: Type.Optional(Type.String({ description: "LibCity task name" })),
			params_file: Type.Optional(Type.String({ description: "Path to a hyperparameter file" })),
			search_space: Type.Optional(
				Type.Object({}, { additionalProperties: true, description: "Hyperparameter search space" }),
			),
			hyper_algo: Type.Optional(Type.String({ description: "Hyperparameter search algorithm" })),
			max_evals: Type.Optional(Type.Number({ description: "Maximum number of evaluations" })),
			max_trials: Type.Optional(Type.Number({ description: "Maximum number of trials" })),
			saved_model: Type.Optional(Type.Boolean({ description: "Whether to save the trained model" })),
			train: Type.Optional(Type.Boolean({ description: "Whether to train after tuning" })),
			gpu: Type.Optional(Type.String({ description: "GPU device to use, e.g. '0' or 'cpu'" })),
			extra_cli_args: Type.Optional(
				Type.Object({}, { additionalProperties: true, description: "Extra CLI arguments as key-value pairs" }),
			),
		}),
		timeoutMs: 7_200_000,
	},
	{
		name: "web_search",
		label: "Web Search",
		description: "Search the web",
		parameters: Type.Object({
			query: Type.String({ description: "Search query" }),
			num: Type.Optional(Type.Number({ description: "Number of results to return" })),
		}),
	},
	{
		name: "web_fetch",
		label: "Web Fetch",
		description: "Fetch a web page and return its text content",
		parameters: Type.Object({
			url: Type.String({ description: "URL of the page to fetch" }),
			max_chars: Type.Optional(Type.Number({ description: "Maximum number of characters to return" })),
		}),
	},
];

interface SubagentDef {
	description: string;
	tools: string[];
	prompt: string;
	model: string;
}

function loadSubagents(): Record<string, SubagentDef> {
	const file = process.env.AGENTCITY_SUBAGENTS_FILE;
	if (!file) return {};
	try {
		return JSON.parse(readFileSync(file, "utf8")) as Record<string, SubagentDef>;
	} catch {
		return {};
	}
}

function taskToolDescription(): string {
	const agents = loadSubagents();
	const names = Object.keys(agents);
	if (names.length === 0) {
		return "Delegate a task to a named sub-agent running in a child pi process. Sub-agent definitions are loaded from the JSON file in AGENTCITY_SUBAGENTS_FILE.";
	}
	const listing = names.map((n) => `- ${n}: ${agents[n].description}`).join("\n");
	return `Delegate a task to a named sub-agent running in a child pi process. Available agents:\n${listing}`;
}

const taskParameters = Type.Object({
	agent: Type.String({ description: "Name of the sub-agent to delegate to" }),
	prompt: Type.String({ description: "The task for the sub-agent" }),
});

export default function (pi: ExtensionAPI) {
	for (const tool of bridgeTools) {
		pi.registerTool({
			name: tool.name,
			label: tool.label,
			description: tool.description,
			parameters: tool.parameters,
			async execute(_toolCallId, params) {
				const text = await runBridgeTool(tool.name, params, tool.timeoutMs ?? DEFAULT_TIMEOUT_MS);
				return { content: [{ type: "text", text }] };
			},
		});
	}

	pi.registerTool({
		name: "task",
		label: "Task",
		description: taskToolDescription(),
		parameters: taskParameters,
		executionMode: "parallel",
		async execute(_toolCallId, params) {
			const agents = loadSubagents();
			const def = agents[params.agent];
			if (!def) {
				const available = Object.keys(agents);
				throw new Error(
					available.length > 0
						? `Unknown sub-agent "${params.agent}". Available agents: ${available.join(", ")}`
						: `Unknown sub-agent "${params.agent}". No sub-agents are defined (check AGENTCITY_SUBAGENTS_FILE).`,
				);
			}

			const piBin = process.env.AGENTCITY_PI_BIN || resolve(extensionDir, "../bin/pi");
			const timeoutMs = Number(process.env.AGENTCITY_TASK_TIMEOUT_MS) || TASK_DEFAULT_TIMEOUT_MS;
			const args = [
				"-p",
				"--no-session",
				"--no-extensions",
				"-e",
				extensionPath,
				"--model",
				def.model,
				"--system-prompt",
				def.prompt,
				"--tools",
				(def.tools ?? []).join(","),
				"--",
				params.prompt,
			];

			const result = await runProcess(piBin, args, { cwd: agentCityCwd(), timeoutMs });

			if (result.timedOut) {
				throw new Error(
					`Sub-agent "${params.agent}" timed out after ${Math.round(timeoutMs / 1000)}s. stderr: ${stderrExcerpt(result.stderr)}`,
				);
			}
			if (result.code !== 0) {
				throw new Error(
					`Sub-agent "${params.agent}" exited with code ${result.code}. stderr: ${stderrExcerpt(result.stderr)}`,
				);
			}

			const output = result.stdout.trim();
			const text =
				output || `Sub-agent "${params.agent}" completed successfully but produced no output.`;
			return { content: [{ type: "text", text }] };
		},
	});
}
