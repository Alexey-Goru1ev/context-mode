/**
 * OpenCode v2 plugin definition for context-mode.
 *
 * OpenCode v2 replaced the v1 plugin API ("V1 plugin implementations do not
 * run in V2" — opencode.ai/v2/docs/build/plugins/migrate-v1). The v2 loader
 * requires a default export with `{ id, setup | effect }`; hook names and
 * event shapes all moved to domain-scoped registrations:
 *
 *   V1 extension point                   → v2 API
 *   ─────────────────────────────────────────────────────────────────────
 *   tool.execute.before                  → ctx.tool.hook("execute.before")
 *   tool.execute.after                   → ctx.tool.hook("execute.after")
 *   event (bus `message.updated`)        → ctx.event.subscribe()
 *   chat.message                         → ctx.session.hook("prompt")
 *   experimental.session.compacting      → ctx.session.hook("compaction")
 *   experimental.chat.system.transform   → ctx.session.hook("context")
 *   tool map (zod args)                  → ctx.tool.transform(editor.add)
 *
 * Shared init (routing, SessionDB, native ctx_* tools) lives in
 * createPluginCore() in ./plugin.ts so both hosts run identical logic.
 *
 * One deliberate semantic mapping: v1's `experimental.session.compacting`
 * exposed a mutable `output.context` array whose entries survived the
 * compaction boundary in the SAME session. The v2 `compaction` hook has no
 * kept-context equivalent (its `system`/`messages` feed the summarization
 * request). Same-session post-compaction continuity therefore flows through
 * an in-memory pending map consumed by the NEXT `context` hook of that
 * session; the durable cross-session path still goes through
 * db.claimLatestUnconsumedResume() exactly as in v1.
 */

import z from "zod/v4";
import {
  extractEvents,
  extractUserEvents,
  parseOpencodeUsage,
  buildAgentUsageEvent,
} from "../../session/extract.js";
import type { HookInput } from "../../session/extract.js";
import { buildResumeSnapshot } from "../../session/snapshot.js";
import type { SessionEvent } from "../../types.js";
import {
  createPluginCore,
  isSyntheticMessage,
  systemHasRoutingInstructions,
} from "./plugin.js";

// ── Types (hand-rolled, host-agnostic — same style as plugin.ts) ──

/**
 * v2 session `context` hook system parts. Verified against
 * @opencode/plugin 2.0.29 dist/promise/session.d.ts
 * (`system: Array<SystemPart>` from @opencode/ai; text parts are
 * `{ type: "text", text: string }`). Plain strings are tolerated so the
 * injection helpers stay defensive against host-side evolution.
 */
type V2SystemPart = { type?: string; text?: string } | string;

/** v2 tool.execute.before event — input is mutable, throw denies the call. */
interface V2ToolExecuteBefore {
  tool: string;
  sessionID: string;
  agent: string;
  messageID: string;
  id: string;
  input: unknown;
}

/** v2 tool.execute.after event — discriminated on `status`. */
type V2ToolExecuteAfter = {
  tool: string;
  sessionID: string;
  input: unknown;
  status: "completed";
  result: { content?: unknown } | undefined;
} | {
  tool: string;
  sessionID: string;
  input: unknown;
  status: "error";
  error: { message?: string } | undefined;
};

/** v2 session `prompt` hook event — `prompt` is an owned mutable draft. */
interface V2SessionPromptEvent {
  sessionID: string;
  prompt?: { text?: string };
}

/** v2 session `context` / `compaction` hook event (fields we consume). */
interface V2SessionContextEvent {
  sessionID: string;
  system?: V2SystemPart[];
}

/** v2 tool executor context (subset we consume). */
interface V2ToolExecuteContext {
  sessionID: string;
  messageID: string;
  agent: string;
  directory?: string;
  signal?: AbortSignal;
  progress?: (update: { status?: string }) => Promise<void>;
}

/** v2 tool definition for editor.add() — input is JSON Schema. */
interface V2ToolDefinition {
  name: string;
  description: string;
  input: Record<string, unknown>;
  execute: (
    input: unknown,
    context: V2ToolExecuteContext,
  ) => Promise<{ content: unknown; metadata?: Record<string, unknown> }>;
}

/**
 * Minimal structural type for the v2 plugin context. The plugin does not
 * depend on @opencode/plugin at runtime (Plugin.define is an identity
 * function; the loader only schema-validates `{ id, setup }`), so these
 * hand-rolled members mirror @opencode/plugin 2.0.29 exactly where used.
 */
export interface V2PluginContext {
  location: { directory?: string };
  tool: {
    transform: (
      callback: (editor: { add: (tool: V2ToolDefinition) => void }) => void,
    ) => Promise<unknown>;
    hook(
      name: "execute.before",
      callback: (event: V2ToolExecuteBefore) => void,
    ): Promise<unknown>;
    hook(
      name: "execute.after",
      callback: (event: V2ToolExecuteAfter) => void,
    ): Promise<unknown>;
  };
  session: {
    hook(
      name: "prompt",
      callback: (event: V2SessionPromptEvent) => void,
    ): Promise<unknown>;
    hook(
      name: "context",
      callback: (event: V2SessionContextEvent) => void,
    ): Promise<unknown>;
    hook(
      name: "compaction",
      callback: (event: V2SessionContextEvent) => void,
    ): Promise<unknown>;
  };
  event: {
    subscribe: (options?: { signal?: AbortSignal }) => AsyncIterable<unknown>;
  };
}

// ── Helpers ───────────────────────────────────────────────

/** Flatten v2 SystemPart entries to plain text for the quorum check. */
function systemPartsText(system: V2SystemPart[]): string[] {
  return system.map((p) => (typeof p === "string" ? p : (p?.text ?? "")));
}

function systemHasRoutingInstructionsV2(system: V2SystemPart[]): boolean {
  return systemHasRoutingInstructions(systemPartsText(system));
}

/** Extract displayable text from a v2 Tool.Result (string or parts array). */
function toolResultText(result: { content?: unknown } | undefined | null): string {
  if (!result) return "";
  const c = result.content;
  if (typeof c === "string") return c;
  if (Array.isArray(c)) {
    return c
      .map((p) =>
        p && typeof p === "object" && typeof (p as { text?: unknown }).text === "string"
          ? (p as { text: string }).text
          : "",
      )
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

/**
 * Convert a native-tool zod args shape (already zod/v4 instances from
 * buildNativeTools → zod3ShapeToV4) into the JSON Schema object the v2
 * tool registry expects. `io: "input"` keeps `.default()` fields optional;
 * `unrepresentable: "any"` degrades exotic constructs instead of throwing.
 */
function zodShapeToJsonSchema(args: Record<string, unknown>): Record<string, unknown> {
  const object = z.object(args as Record<string, z.ZodType>);
  return z.toJSONSchema(object, {
    io: "input",
    unrepresentable: "any",
  }) as Record<string, unknown>;
}

// ── V2 setup factory ──────────────────────────────────────

/**
 * Build the v2 `setup(ctx)` function. All shared initialization is
 * delegated to createPluginCore(); this file only owns v2 wiring.
 */
export function createV2Setup() {
  return async function setupOpenCodeV2(ctx: V2PluginContext): Promise<() => void> {
    const core = await createPluginCore({
      directory: ctx.location?.directory ?? process.cwd(),
      log: async (message, extra) => {
        // v2 has no client.app.log domain; the server captures stdout.
        console.log(
          `[context-mode] ${message ?? ""}${extra ? ` ${JSON.stringify(extra)}` : ""}`,
        );
      },
    });

    const { platform, routing, routingBlock, autoInjectionMod, db, projectDir, captureAgentsMd, safeLog, nativeTools } =
      core;

    // Pending same-session post-compaction injections. Keyed by sessionId,
    // produced by the `compaction` hook, consumed exactly once by the next
    // `context` hook of the same session (see module docstring).
    const pendingCompactionInjections = new Map<string, { snapshot: string; autoBlock?: string }>();

    // ── Native ctx_* tools (v2 tool registry, JSON Schema input) ──

    await ctx.tool.transform((editor) => {
      for (const [name, def] of Object.entries(nativeTools)) {
        let input: Record<string, unknown>;
        try {
          input = zodShapeToJsonSchema(def.args as Record<string, unknown>);
        } catch (err) {
          // Fail loud per-tool, but never kill the whole plugin load.
          const message = err instanceof Error ? err.message : String(err);
          console.error(`[context-mode] JSON Schema conversion failed for ${name}: ${message}`);
          void safeLog(`JSON Schema conversion failed for ${name}: ${message}`);
          input = { type: "object", additionalProperties: true };
        }
        editor.add({
          name,
          description: def.description,
          input,
          async execute(inputArgs, context) {
            // NativeToolDefinition.execute handles zod parsing, project-dir
            // override, MCP result flattening, and isError → throw (#574/#621).
            const result = await def.execute(inputArgs as Record<string, unknown>, {
              sessionID: context.sessionID,
              messageID: context.messageID,
              agent: context.agent,
              directory: context.directory ?? projectDir,
              // v1 toolCtx.metadata({title}) → v2 progress({status}).
              metadata: (m) => context.progress?.({ status: m?.title }),
            });
            const output = typeof result === "string" ? result : result.output;
            const title =
              typeof result === "string" ? name : (result.title ?? name);
            return {
              content: output,
              metadata: { title },
            };
          },
        });
      }
    });

    // ── tool.execute.before: Routing enforcement ─────────────

    await ctx.tool.hook("execute.before", (event) => {
      const toolName = event.tool ?? "";
      const toolInput =
        event.input && typeof event.input === "object"
          ? (event.input as Record<string, unknown>)
          : {};

      let decision;
      try {
        decision = routing.routePreToolUse(toolName, toolInput, projectDir, platform);
      } catch {
        return; // Routing failure → allow passthrough
      }

      if (!decision) return; // No routing match → passthrough

      if (decision.action === "deny" || decision.action === "ask") {
        // Throw to block — v2 denies the tool call (same contract as v1).
        throw new Error(decision.reason ?? "Blocked by context-mode");
      }

      // Mutating event.input requires an object; replace it outright when
      // the host delivered a non-object input alongside a modify/context
      // decision (v1 always mutated output.args in place).
      if (!(event.input && typeof event.input === "object")) {
        if (decision.action === "modify" && decision.updatedInput) {
          event.input = { ...decision.updatedInput };
        } else if (decision.action === "context" && decision.additionalContext) {
          event.input = { additionalContext: decision.additionalContext };
        }
        return;
      }

      if (decision.action === "modify" && decision.updatedInput) {
        Object.assign(toolInput, decision.updatedInput);
      }

      if (decision.action === "context" && decision.additionalContext) {
        toolInput.additionalContext = decision.additionalContext;
      }
    });

    // ── tool.execute.after: Session event capture ───────────

    await ctx.tool.hook("execute.after", (event) => {
      const sessionId = event.sessionID;
      if (!sessionId) return;
      try {
        db.ensureSession(sessionId, projectDir);
        // OC-4 (#487 follow-up): AGENTS.md → rule_content capture.
        captureAgentsMd(sessionId);

        const hookInput: HookInput = {
          tool_name: event.tool ?? "",
          tool_input:
            event.input && typeof event.input === "object"
              ? (event.input as Record<string, unknown>)
              : {},
          tool_response:
            event.status === "completed"
              ? toolResultText(event.result)
              : event.status === "error"
                ? (event.error?.message ?? "")
                : "",
          tool_output: undefined, // status is carried in tool_response text
        };

        const events = extractEvents(hookInput);
        for (const ev of events) {
          db.insertEvent(sessionId, ev as SessionEvent, "PostToolUse");
        }
      } catch {
        // Silent — session capture must never break the tool call
      }
    });

    // ── session prompt: User-prompt capture (OC-2 / Z2) ──────

    await ctx.session.hook("prompt", (event) => {
      const sessionId = event.sessionID;
      if (!sessionId) return;
      try {
        const message = event.prompt?.text;
        if (!message) return;
        // CCv2 inline filter: skip synthetic harness messages so we don't
        // pollute the user-prompt event stream.
        if (isSyntheticMessage(message)) return;

        db.ensureSession(sessionId, projectDir);
        captureAgentsMd(sessionId);

        db.insertEvent(sessionId, {
          type: "user_prompt",
          category: "user-prompt",
          data: message,
          priority: 1,
        } as SessionEvent, "UserPromptSubmit");

        const userEvents = extractUserEvents(message);
        for (const ev of userEvents) {
          db.insertEvent(sessionId, ev as SessionEvent, "UserPromptSubmit");
        }
      } catch {
        // Silent — prompt capture must never break admission
      }
    });

    // ── session compaction: Snapshot generation (OC-3) ───────

    await ctx.session.hook("compaction", async (event) => {
      const sessionId = event.sessionID;
      if (!sessionId) return;
      try {
        db.ensureSession(sessionId, projectDir);
        const events = db.getEvents(sessionId);
        if (events.length === 0) return;

        const stats = db.getSessionStats(sessionId);
        const snapshot = buildResumeSnapshot(events, {
          compactCount: (stats?.compact_count ?? 0) + 1,
        });

        db.upsertResume(sessionId, snapshot, events.length);
        db.incrementCompactCount(sessionId);

        // Budget-capped auto-injection block (P1 role / P2 rules / P3
        // skills / P4 intent — ≤500 tokens / ~2000 chars per
        // hooks/auto-injection.mjs). Kept as a separate entry so it folds
        // independently from the verbose snapshot.
        let autoBlock: string | undefined;
        try {
          const block: string = autoInjectionMod.buildAutoInjection(events);
          if (block && block.length > 0) autoBlock = block;
        } catch {
          // Auto-injection failure must NOT break the snapshot path.
        }

        pendingCompactionInjections.set(sessionId, { snapshot, autoBlock });

        if (process.env.OPENCODE_DEBUG) {
          await safeLog(snapshot, {
            sessionId,
            source: "on compaction - snapshot",
          });
        }
      } catch {
        // Silent — never break the compaction flow
      }
    });

    // ── session context: ROUTING_BLOCK + resume injection (OC-1) ──

    await ctx.session.hook("context", (event) => {
      const sessionId = event.sessionID;
      if (!sessionId) return;
      const system = event.system;
      if (!Array.isArray(system)) return;

      // ROUTING_BLOCK injection on the first context hook per session.
      // Skip when the system prompt already contains context-mode routing
      // rules (quorum of distinctive tool names, #487).
      if (!systemHasRoutingInstructionsV2(system)) {
        try {
          // Splice at index 1 (NOT unshift): keeps system[0] (the host
          // header part) intact so provider prompt-cache folds apply the
          // same way as the v1 llm.ts:117-128 header invariant.
          system.splice(1, 0, { type: "text", text: routingBlock });
        } catch {
          // Never break the model request on injection failure.
        }

        if (process.env.OPENCODE_DEBUG) {
          void safeLog(routingBlock.slice(0, 200), { sessionId, source: "on routing block injection" });
        }
      } else if (process.env.OPENCODE_DEBUG) {
        void safeLog(
          "routing block skipped — system prompt already contains context-mode instructions",
          { sessionId, source: "on routing block injection" },
        );
      }

      // Same-session post-compaction continuity (v2 replacement for the v1
      // compacting hook's mutable output.context — see module docstring).
      const pending = pendingCompactionInjections.get(sessionId);
      if (pending) {
        pendingCompactionInjections.delete(sessionId);
        try {
          system.splice(1, 0, { type: "text", text: pending.snapshot });
          if (pending.autoBlock) {
            system.splice(2, 0, { type: "text", text: pending.autoBlock });
          }
        } catch {
          // Never break the model request on injection failure.
        }
        return;
      }

      // Cross-session resume: atomically claim the most-recent unconsumed
      // snapshot from a DIFFERENT session (SQL excludes self — v1.0.106
      // Mickey #376 follow-up semantics preserved).
      try {
        const row = db.claimLatestUnconsumedResume(sessionId);
        if (!row || !row.snapshot) return; // no row → retry on next turn

        system.splice(1, 0, { type: "text", text: row.snapshot });

        if (process.env.OPENCODE_DEBUG) {
          void safeLog(row.snapshot.slice(0, 200), { sessionId, source: "on resume" });
        }
      } catch {
        // Silent — never break the chat turn
      }
    });

    // ── event stream: per-turn token + cost capture ─────────

    const controller = new AbortController();
    void (async () => {
      try {
        for await (const ev of ctx.event.subscribe({ signal: controller.signal })) {
          try {
            // Same event filter as the v1 bus `event` hook. The v2 stream
            // delivers { type, properties } encoded events; extract.ts's
            // parseOpencodeUsage() already unwraps every historical shape.
            const e = ev as {
              type?: string;
              properties?: { info?: { sessionID?: string } };
            };
            if (!e || e.type !== "message.updated") continue;
            const sessionId = e.properties?.info?.sessionID;
            if (!sessionId || typeof sessionId !== "string") continue;

            const counts = parseOpencodeUsage(ev);
            if (!counts) continue;
            const usageEvent = buildAgentUsageEvent(counts);
            if (!usageEvent) continue;

            db.ensureSession(sessionId, projectDir);
            db.insertEvent(sessionId, usageEvent, "MessageUpdated");
          } catch {
            // Silent — usage capture must never break the session.
          }
        }
      } catch {
        // Stream aborted during cleanup or transport closed — expected.
      }
    })();

    // ── Cleanup ─────────────────────────────────────────────

    return () => {
      controller.abort();
      pendingCompactionInjections.clear();
    };
  };
}
