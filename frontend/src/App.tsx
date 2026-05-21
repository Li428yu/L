import { Fragment, useEffect, useMemo, useState } from "react";
import type { CSSProperties, MouseEvent, ReactNode } from "react";
import {
  askPaperStream,
  deleteConversation,
  deleteDocument,
  documentFileUrl,
  getConversation,
  getModels,
  getTask,
  listConversations,
  listDocuments,
  uploadDocument
} from "./api";
import type {
  AskResponse,
  ChatMessage,
  ConversationDetail,
  DocumentInfo,
  EvidenceItem,
  ModelCatalog,
  RagTrace,
  RuntimeStep,
  TaskInfo
} from "./types";

interface ConversationDraft {
  id: string;
  title: string;
  updatedAt: string;
  conversationId: string | null;
  messages: ChatMessage[];
  activeEvidence: EvidenceItem | null;
  trace: RagTrace | null;
}

interface PendingUpload {
  id: string;
  name: string;
}

interface ConversationContextMenu {
  conversationId: string;
  x: number;
  y: number;
}

interface StoredModelSettings {
  modelPreset?: string;
  chatModel?: string;
  embeddingModel?: string;
  topK?: number;
}

const starterQuestions = [
  "请分别概括每篇文档讲了什么。",
  "请分别说出每篇文档最重要的发现。",
  "请分别说明每篇文档用了什么方法。",
  "请分别说说每篇文档有什么局限。"
];

const ACTIVE_CONVERSATION_STORAGE_KEY = "paper-reading-assistant.active-conversation-id";
const DELETED_CONVERSATIONS_STORAGE_KEY = "paper-reading-assistant.deleted-conversation-ids";
const MODEL_SETTINGS_STORAGE_KEY = "paper-reading-assistant.model-settings";

function readStoredModelSettings(): StoredModelSettings {
  try {
    const raw = window.localStorage.getItem(MODEL_SETTINGS_STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw) as StoredModelSettings;
    return typeof parsed === "object" && parsed ? parsed : {};
  } catch {
    return {};
  }
}

function clampTopK(value: number): number {
  if (!Number.isFinite(value)) {
    return 5;
  }
  return Math.min(Math.max(Math.round(value), 1), 20);
}

function uniqueNonEmpty(values: Array<string | null | undefined>): string[] {
  return Array.from(new Set(values.map((value) => value?.trim()).filter((value): value is string => Boolean(value))));
}

function deletedConversationIds(): string[] {
  try {
    const raw = window.localStorage.getItem(DELETED_CONVERSATIONS_STORAGE_KEY);
    const values = raw ? (JSON.parse(raw) as unknown) : [];
    return Array.isArray(values) ? values.filter((value): value is string => typeof value === "string") : [];
  } catch {
    return [];
  }
}

function rememberDeletedConversationId(conversationId: string | null | undefined) {
  if (!conversationId) {
    return;
  }
  const next = Array.from(new Set([...deletedConversationIds(), conversationId])).slice(-200);
  window.localStorage.setItem(DELETED_CONVERSATIONS_STORAGE_KEY, JSON.stringify(next));
}

function newConversation(): ConversationDraft {
  return {
    id: crypto.randomUUID(),
    title: "新的论文对话",
    updatedAt: new Date().toISOString(),
    conversationId: null,
    messages: [],
    activeEvidence: null,
    trace: null
  };
}

function conversationFromDetail(detail: ConversationDetail): ConversationDraft {
  const messages: ChatMessage[] = detail.messages.map((message) => ({
    id: String(message.id),
    role: message.role,
    content: message.content,
    evidence: message.evidence,
    memory_used: message.role === "assistant" ? detail.memory_used : undefined
  }));
  const latestAssistant = [...messages].reverse().find((message) => message.role === "assistant");
  return {
    id: detail.conversation.id,
    title: detail.conversation.title || "历史对话",
    updatedAt: detail.conversation.updated_at,
    conversationId: detail.conversation.id,
    messages,
    activeEvidence: latestAssistant?.evidence?.[0] ?? null,
    trace: null
  };
}

function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [models, setModels] = useState<ModelCatalog | null>(null);
  const [modelPreset, setModelPreset] = useState("balanced");
  const [chatModel, setChatModel] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [topK, setTopK] = useState(5);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [task, setTask] = useState<TaskInfo | null>(null);
  const [conversations, setConversations] = useState<ConversationDraft[]>([newConversation()]);
  const [activeConversationId, setActiveConversationId] = useState("");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [rightPanelTab, setRightPanelTab] = useState<"evidence" | "teaching">("evidence");
  const [pendingUploads, setPendingUploads] = useState<PendingUpload[]>([]);
  const [deletingDocumentIds, setDeletingDocumentIds] = useState<string[]>([]);
  const [columnWidths, setColumnWidths] = useState({ left: 280, right: 420 });
  const [conversationMenu, setConversationMenu] = useState<ConversationContextMenu | null>(null);

  const readyDocuments = documents.filter((document) => document.status === "ready");
  const preparingDocuments = documents.filter(
    (document) => document.status === "queued" || document.status === "processing"
  );
  const failedDocuments = documents.filter((document) => document.status === "failed");
  const taskStillRunning = Boolean(task && (task.status === "queued" || task.status === "running"));
  const canAsk =
    readyDocuments.length > 0 &&
    preparingDocuments.length === 0 &&
    pendingUploads.length === 0 &&
    !taskStillRunning;

  const activeConversation = useMemo(() => {
    const fallback = conversations[0];
    return conversations.find((item) => item.id === activeConversationId) || fallback;
  }, [activeConversationId, conversations]);

  useEffect(() => {
    setActiveConversationId((current) => current || conversations[0].id);
  }, [conversations]);

  useEffect(() => {
    refreshDocuments();
    loadConversationHistory();
    getModels()
      .then((catalog) => {
        const storedSettings = readStoredModelSettings();
        const preset = catalog.presets.find((item) => item.id === storedSettings.modelPreset) ||
          catalog.presets.find((item) => item.id === catalog.default_preset);
        const nextChatModel = catalog.chat_model_options.includes(storedSettings.chatModel || "")
          ? storedSettings.chatModel || ""
          : preset?.chat_model || catalog.default_chat_model;
        const nextEmbeddingModel = catalog.embedding_model_options.includes(storedSettings.embeddingModel || "")
          ? storedSettings.embeddingModel || ""
          : preset?.embedding_model || catalog.default_embedding_model;
        setModels(catalog);
        setModelPreset(preset?.id || catalog.default_preset);
        setChatModel(nextChatModel);
        setEmbeddingModel(nextEmbeddingModel);
        setTopK(clampTopK(storedSettings.topK ?? preset?.top_k ?? catalog.default_top_k));
      })
      .catch((error: Error) => setNotice(toFriendlyError(error.message)));
  }, []);

  useEffect(() => {
    if (!taskStillRunning && preparingDocuments.length === 0) {
      return;
    }
    const timer = window.setInterval(() => {
      if (taskStillRunning && task) {
        getTask(task.id)
          .then((nextTask) => {
            setTask(nextTask);
            if (nextTask.status === "completed" || nextTask.status === "failed") {
              refreshDocuments();
            }
          })
          .catch((error: Error) => setNotice(toFriendlyError(error.message)));
      }
      refreshDocuments();
    }, 1200);
    return () => window.clearInterval(timer);
  }, [task, taskStillRunning, preparingDocuments.length]);

  useEffect(() => {
    if (activeConversation?.conversationId) {
      window.localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, activeConversation.conversationId);
    }
  }, [activeConversation?.conversationId, activeConversation?.id]);

  useEffect(() => {
    if (!models) {
      return;
    }
    window.localStorage.setItem(
      MODEL_SETTINGS_STORAGE_KEY,
      JSON.stringify({
        modelPreset,
        chatModel,
        embeddingModel,
        topK
      })
    );
  }, [models, modelPreset, chatModel, embeddingModel, topK]);

  useEffect(() => {
    if (!conversationMenu) {
      return;
    }
    function closeMenu() {
      setConversationMenu(null);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        closeMenu();
      }
    }
    window.addEventListener("click", closeMenu);
    window.addEventListener("scroll", closeMenu, true);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("click", closeMenu);
      window.removeEventListener("scroll", closeMenu, true);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [conversationMenu]);

  function updateActiveConversation(patch: Partial<ConversationDraft>) {
    setConversations((current) =>
      current.map((item) =>
        item.id === activeConversation.id
          ? { ...item, ...patch, updatedAt: new Date().toISOString() }
          : item
      )
    );
  }

  async function loadConversationHistory() {
    try {
      const summaries = await listConversations();
      const deletedIds = new Set(deletedConversationIds());
      const visibleSummaries = summaries.filter((conversation) => !deletedIds.has(conversation.id));
      if (!visibleSummaries.length) {
        return;
      }
      const details = await Promise.all(
        visibleSummaries.slice(0, 20).map((conversation) =>
          getConversation(conversation.id).catch(() => null)
        )
      );
      const restored = details
        .filter((detail): detail is ConversationDetail => Boolean(detail))
        .filter((detail) => !deletedIds.has(detail.conversation.id))
        .map(conversationFromDetail)
        .filter((conversation) => conversation.messages.length > 0);
      if (!restored.length) {
        return;
      }
      setConversations(restored);
      const savedId = window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY);
      const restoredActive = restored.find(
        (conversation) => conversation.id === savedId || conversation.conversationId === savedId
      );
      setActiveConversationId(restoredActive?.id || restored[0].id);
    } catch (error) {
      setNotice(error instanceof Error ? toFriendlyError(error.message) : "历史对话恢复失败，但不影响继续提问。");
    }
  }

  function refreshDocuments() {
    listDocuments()
      .then((items) => {
        setDocuments(items);
      })
      .catch((error: Error) => setNotice(toFriendlyError(error.message)));
  }

  async function handleUpload(files: File[]) {
    if (files.length === 0) {
      return;
    }
    setNotice("");
    const uploads = files.map((file) => ({
      id: crypto.randomUUID(),
      name: file.name
    }));
    setPendingUploads((current) => [...current, ...uploads]);

    for (const [index, file] of files.entries()) {
      try {
        const response = await uploadDocument(file);
        setTask(response.task);
        setNotice(`已收到《${response.document.file_name}》。等小圆圈消失后，就可以直接提问。`);
        refreshDocuments();
      } catch (error) {
        const reason = error instanceof Error ? toFriendlyError(error.message) : "上传失败，请稍后再试。";
        setNotice(`《${file.name}》上传失败：${reason}`);
      } finally {
        setPendingUploads((current) => current.filter((item) => item.id !== uploads[index].id));
      }
    }
    refreshDocuments();
  }

  async function handleDeleteDocument(document: DocumentInfo) {
    const confirmed = window.confirm(`删除《${document.file_name}》后，它不会再参与后续回答，确定删除吗？`);
    if (!confirmed) {
      return;
    }

    setNotice("");
    setDeletingDocumentIds((current) => [...current, document.id]);
    try {
      await deleteDocument(document.id);
      setDocuments((current) => current.filter((item) => item.id !== document.id));
      if (task?.document_id === document.id) {
        setTask(null);
      }
      if (activeConversation.activeEvidence?.document_id === document.id) {
        updateActiveConversation({ activeEvidence: null });
      }
      setNotice(`已删除《${document.file_name}》，后续回答不会再使用它。`);
    } catch (error) {
      setNotice(error instanceof Error ? toFriendlyError(error.message) : "删除失败，请稍后再试。");
      refreshDocuments();
    } finally {
      setDeletingDocumentIds((current) => current.filter((id) => id !== document.id));
    }
  }

  function applyModelPreset(presetId: string) {
    setModelPreset(presetId);
    const preset = models?.presets.find((item) => item.id === presetId);
    if (!preset) {
      return;
    }
    setChatModel(preset.chat_model);
    setEmbeddingModel(preset.embedding_model);
    setTopK(clampTopK(preset.top_k));
  }

  function resetModelSettings() {
    if (!models) {
      return;
    }
    applyModelPreset(models.default_preset);
  }

  async function handleAsk() {
    const cleanQuestion = question.trim();
    if (!cleanQuestion) {
      setNotice("先输入一个问题，我再帮你读文档。");
      return;
    }
    if (pendingUploads.length > 0 || preparingDocuments.length > 0 || taskStillRunning) {
      setNotice("文档还在上传或准备中，等小圆圈消失后再提问。");
      return;
    }
    if (readyDocuments.length === 0) {
      setNotice(
        failedDocuments.length > 0
          ? "文档准备失败了，请重新上传一份清晰的 PDF 或 DOCX。"
          : "请先点击 + 上传 PDF 或 DOCX。"
      );
      return;
    }

    setBusy(true);
    setNotice("");
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: cleanQuestion
    };
    const assistantId = crypto.randomUUID();
    const conversationLocalId = activeConversation.id;
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "正在理解你的问题..."
    };
    const nextMessages = [...activeConversation.messages, userMessage, assistantMessage];
    updateActiveConversation({
      messages: nextMessages,
      title: activeConversation.messages.length === 0 ? cleanQuestion.slice(0, 24) : activeConversation.title
    });
    setQuestion("");

    const updateStreamingMessage = (patch: Partial<ChatMessage>) => {
      setConversations((current) =>
        current.map((conversation) =>
          conversation.id === conversationLocalId
            ? {
                ...conversation,
                updatedAt: new Date().toISOString(),
                messages: conversation.messages.map((message) =>
                  message.id === assistantId ? { ...message, ...patch } : message
                )
              }
            : conversation
        )
      );
    };

    try {
      let streamedAnswer = "";
      let answerStarted = false;
      const response: AskResponse = await askPaperStream(
        {
          question: cleanQuestion,
          conversation_id: activeConversation.conversationId,
          document_ids: readyDocuments.map((document) => document.id),
          model_preset: modelPreset,
          chat_model: chatModel || undefined,
          embedding_model: embeddingModel || undefined,
          top_k: topK
        },
        {
          onStatus: (status) => {
            if (!answerStarted) {
              updateStreamingMessage({ content: status });
            }
          },
          onChunk: (chunk) => {
            if (!answerStarted) {
              answerStarted = true;
              streamedAnswer = "";
            }
            streamedAnswer += chunk;
            updateStreamingMessage({ content: streamedAnswer });
          }
        }
      );
      setConversations((current) =>
        current.map((conversation) =>
          conversation.id === conversationLocalId
            ? {
                ...conversation,
                conversationId: response.conversation_id,
                activeEvidence: response.evidence[0] ?? null,
                trace: response.rag_trace,
                updatedAt: new Date().toISOString(),
                messages: conversation.messages.map((message) =>
                  message.id === assistantId
                    ? {
                        ...message,
                        content: streamedAnswer || response.answer,
                        evidence: response.evidence,
                        runtime: response.runtime,
                        rag_trace: response.rag_trace,
                        memory_used: response.memory_used
                      }
                    : message
                )
              }
            : conversation
        )
      );
    } catch (error) {
      updateStreamingMessage({
        content: "这次回答没有成功生成，请稍后再试。"
      });
      setNotice(error instanceof Error ? toFriendlyError(error.message) : "提问失败，请稍后再试。");
    } finally {
      setBusy(false);
    }
  }

  function createConversation() {
    const draft = newConversation();
    setConversations((current) => [draft, ...current]);
    setActiveConversationId(draft.id);
    setQuestion("");
    setConversationMenu(null);
  }

  async function handleDeleteConversation(conversation: ConversationDraft) {
    setConversationMenu(null);
    if (busy && conversation.id === activeConversation.id) {
      setNotice("当前对话正在生成回答，等回答完成后再删除。");
      return;
    }
    const confirmed = window.confirm(
      `确定删除《${conversation.title}》这条历史对话吗？\n\n只会删除对话记录和这条对话里的长期记忆，不会删除已上传文档。`
    );
    if (!confirmed) {
      return;
    }

    const replacement = newConversation();
    const remaining = conversations.filter((item) => item.id !== conversation.id);
    const nextConversations = remaining.length ? remaining : [replacement];
    const nextActiveId =
      conversation.id === activeConversation.id
        ? nextConversations[0].id
        : activeConversation.id;

    try {
      const backendConversationId =
        conversation.conversationId || (conversation.id.startsWith("conv_") ? conversation.id : null);
      if (backendConversationId) {
        await deleteConversation(backendConversationId);
        const afterDelete = await listConversations();
        if (afterDelete.some((item) => item.id === backendConversationId)) {
          throw new Error("后端仍然返回这条对话，数据库没有真正删除。请确认后端服务已经重启到最新版本。");
        }
        rememberDeletedConversationId(backendConversationId);
      } else {
        // A conversation without a backend id is only a local draft, so removing
        // it from the current UI is enough.
        rememberDeletedConversationId(conversation.id);
      }
      setConversations(nextConversations);
      setActiveConversationId(nextActiveId);
      const nextActiveConversation = nextConversations.find((item) => item.id === nextActiveId);
      const savedId = window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY);
      if (savedId === backendConversationId || savedId === conversation.id || conversation.id === activeConversation.id) {
        if (nextActiveConversation?.conversationId) {
          window.localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, nextActiveConversation.conversationId);
        } else {
          window.localStorage.removeItem(ACTIVE_CONVERSATION_STORAGE_KEY);
        }
      }
      setNotice("已删除这条历史对话，上传的文档不会受影响。");
    } catch (error) {
      setNotice(
        error instanceof Error
          ? `删除失败：${toFriendlyError(error.message)}`
          : "删除失败：后端没有确认删除这条历史对话。"
      );
    }
  }

  function openConversationMenu(event: MouseEvent, conversation: ConversationDraft) {
    event.preventDefault();
    event.stopPropagation();
    const menuWidth = 178;
    const menuHeight = 52;
    setConversationMenu({
      conversationId: conversation.id,
      x: Math.min(event.clientX, window.innerWidth - menuWidth - 10),
      y: Math.min(event.clientY, window.innerHeight - menuHeight - 10)
    });
  }

  function pickEvidence(item: EvidenceItem) {
    const documentExists = documents.some((document) => document.id === item.document_id);
    if (!documentExists) {
      setNotice("这篇文档已删除，原文证据不再可用。");
      return;
    }
    setRightPanelTab("evidence");
    updateActiveConversation({ activeEvidence: item });
  }

  function startColumnResize(side: "left" | "right") {
    const startWidths = { ...columnWidths };
    const centerMinWidth = 460;
    const leftMinWidth = 180;
    const rightMinWidth = 320;
    const handleWidth = 16;

    function clamp(value: number, min: number, max: number) {
      return Math.min(Math.max(value, min), Math.max(min, max));
    }

    function handlePointerMove(event: PointerEvent) {
      const viewportWidth = window.innerWidth;
      if (side === "left") {
        const maxLeftWidth = viewportWidth - startWidths.right - centerMinWidth - handleWidth;
        setColumnWidths((current) => ({
          ...current,
          left: clamp(event.clientX, leftMinWidth, maxLeftWidth)
        }));
        return;
      }

      const maxRightWidth = viewportWidth - startWidths.left - centerMinWidth - handleWidth;
      setColumnWidths((current) => ({
        ...current,
        right: clamp(viewportWidth - event.clientX, rightMinWidth, maxRightWidth)
      }));
    }

    function stopResize() {
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopResize);
    }

    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopResize, { once: true });
  }

  const activePreset = models?.presets.find((item) => item.id === modelPreset);
  const chatModelOptions = uniqueNonEmpty(models?.chat_model_options || [chatModel]);
  const embeddingModelOptions = uniqueNonEmpty(models?.embedding_model_options || [embeddingModel]);
  const latestAssistantMessage = [...activeConversation.messages]
    .reverse()
    .find((message) => message.role === "assistant");
  const latestUserQuestion =
    [...activeConversation.messages].reverse().find((message) => message.role === "user")?.content || "";
  const latestRuntime = latestAssistantMessage?.runtime || [];
  const latestTrace = latestAssistantMessage?.rag_trace || activeConversation.trace;
  const menuConversation = conversationMenu
    ? conversations.find((conversation) => conversation.id === conversationMenu.conversationId)
    : null;
  const shellStyle = {
    "--history-width": `${columnWidths.left}px`,
    "--source-width": `${columnWidths.right}px`
  } as CSSProperties;

  return (
    <main className="app-shell" style={shellStyle}>
      <aside className="history-rail">
        <div className="brand">
          <span>论文阅读助手</span>
          <button onClick={createConversation}>新对话</button>
        </div>
        <div className="history-list">
          {conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={conversation.id === activeConversation.id ? "history-item active" : "history-item"}
              onContextMenu={(event) => openConversationMenu(event, conversation)}
            >
              <button className="history-open" onClick={() => setActiveConversationId(conversation.id)}>
                <strong>{conversation.title}</strong>
                <span>{conversation.messages.length ? `${conversation.messages.length} 条消息` : "还没开始"}</span>
              </button>
            </div>
          ))}
        </div>
      </aside>

      {conversationMenu && menuConversation && (
        <div
          className="history-context-menu"
          style={{ left: conversationMenu.x, top: conversationMenu.y }}
          onClick={(event) => event.stopPropagation()}
        >
          <button className="context-danger" onClick={() => handleDeleteConversation(menuConversation)}>
            删除这条对话
          </button>
        </div>
      )}

      {settingsOpen && (
        <SettingsPanel
          catalog={models}
          modelPreset={modelPreset}
          chatModel={chatModel}
          embeddingModel={embeddingModel}
          topK={topK}
          chatModelOptions={chatModelOptions}
          embeddingModelOptions={embeddingModelOptions}
          onClose={() => setSettingsOpen(false)}
          onPresetChange={applyModelPreset}
          onChatModelChange={setChatModel}
          onEmbeddingModelChange={setEmbeddingModel}
          onTopKChange={(value) => setTopK(clampTopK(value))}
          onReset={resetModelSettings}
        />
      )}

      <button
        className="column-resizer"
        type="button"
        aria-label="调整左侧历史栏宽度"
        title="拖动调整历史栏宽度"
        onPointerDown={(event) => {
          event.preventDefault();
          startColumnResize("left");
        }}
      />

      <section className="chat-column">
        <header className="top-bar">
          <div>
            <h1>和文档对话</h1>
            <p>上传 PDF 或 DOCX，然后像聊天一样提问。我会在右侧给出原文证据。</p>
          </div>
        </header>

        {notice && <div className="notice">{notice}</div>}

        <section className="chat-stream">
          {activeConversation.messages.length === 0 && (
            <div className="empty-chat">
              <h2>可以这样问我</h2>
              <div className="starter-grid">
                {starterQuestions.map((item) => (
                  <button key={item} onClick={() => setQuestion(item)}>
                    {item}
                  </button>
                ))}
              </div>
              {activePreset && <p>当前是「{activePreset.label}」：{activePreset.description}</p>}
            </div>
          )}

          {activeConversation.messages.map((message) => (
            <article key={message.id} className={`message-row ${message.role}`}>
              <div className="avatar">{message.role === "user" ? "你" : "助"}</div>
              <div className="bubble">
                {message.role === "assistant"
                  ? renderAnswerWithCitations(
                      message.content,
                      message.evidence || [],
                      pickEvidence
                    )
                  : message.content}
                {message.role === "assistant" && renderEvidenceActions(message, pickEvidence)}
              </div>
            </article>
          ))}
          {busy && activeConversation.messages[activeConversation.messages.length - 1]?.role !== "assistant" && (
            <article className="message-row assistant">
              <div className="avatar">助</div>
              <div className="bubble thinking">我正在查找原文证据并整理回答...</div>
            </article>
          )}
        </section>

        <footer className="composer">
          <div className="attachment-tray">
            {pendingUploads.map((upload) => (
              <span className="uploading-chip" key={upload.id}>
                <span className="spinner" />
                正在上传：{upload.name}
              </span>
            ))}
            {taskStillRunning && pendingUploads.length === 0 && preparingDocuments.length === 0 && (
              <span className="uploading-chip">
                <span className="spinner" />
                文档准备中
              </span>
            )}
            {readyDocuments.length > 0 && (
              <span className="attachment-hint">上传成功的文档都会参与回答</span>
            )}
            {documents.length ? (
              documents.map((document) => (
                <span key={document.id} className={`attachment-chip ${document.status}`}>
                  {(document.status === "queued" || document.status === "processing") && (
                    <span className="spinner" />
                  )}
                  {document.file_name}
                  {document.status !== "ready" ? ` · ${humanDocumentStatus(document.status)}` : ""}
                  <button
                    className="delete-chip"
                    type="button"
                    aria-label={`删除 ${document.file_name}`}
                    title="删除这篇文档"
                    disabled={deletingDocumentIds.includes(document.id)}
                    onClick={() => handleDeleteDocument(document)}
                  >
                    ×
                  </button>
                </span>
              ))
            ) : (
              <span className="attachment-hint">点击 + 上传 PDF 或 DOCX</span>
            )}
          </div>
          <div className="composer-row">
            <label className="attach-button" title="上传 PDF 或 DOCX">
              <input
                type="file"
                accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                multiple
                onChange={(event) => {
                  handleUpload(Array.from(event.currentTarget.files || []));
                  event.currentTarget.value = "";
                }}
              />
              +
            </label>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={
                canAsk ? "直接输入你的问题，例如：这篇论文的结论可靠吗？" : "先上传文档，等小圆圈消失后就能提问"
              }
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  handleAsk();
                }
              }}
            />
            <button onClick={handleAsk} disabled={busy || !canAsk}>
              {busy ? "阅读中" : "发送"}
            </button>
          </div>
        </footer>
      </section>

      <button
        className="column-resizer"
        type="button"
        aria-label="调整右侧面板宽度"
        title="拖动调整右侧面板宽度"
        onPointerDown={(event) => {
          event.preventDefault();
          startColumnResize("right");
        }}
      />

      <aside className="source-column">
        <div className="right-tabs">
          <button
            className={rightPanelTab === "evidence" ? "active" : ""}
            onClick={() => setRightPanelTab("evidence")}
          >
            原文证据
          </button>
          <button
            className={rightPanelTab === "teaching" ? "active" : ""}
            onClick={() => setRightPanelTab("teaching")}
          >
            教学观察
          </button>
        </div>
        {rightPanelTab === "evidence" ? (
          <EvidenceViewer evidence={activeConversation.activeEvidence} documents={documents} />
        ) : (
          <TeachingPanel
            documents={documents}
            runtime={latestRuntime}
            trace={latestTrace}
            latestQuestion={latestUserQuestion}
            memoryUsed={latestAssistantMessage?.memory_used || {}}
          />
        )}
      </aside>
    </main>
  );
}

function EvidenceViewer({
  evidence,
  documents
}: {
  evidence: EvidenceItem | null;
  documents: DocumentInfo[];
}) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const evidenceKey = evidence ? `${evidence.document_id}:${evidence.chunk_id}:${evidence.page}` : "";

  useEffect(() => {
    setPreviewOpen(false);
  }, [evidenceKey]);

  if (!evidence) {
    return (
      <section className="source-card empty-source">
        <h2>原文证据</h2>
        <p>回答里出现“证据 1”“证据 2”时，点击它们，我会在这里展示对应的原文段落。</p>
      </section>
    );
  }

  const document = documents.find((item) => item.id === evidence.document_id);
  const isPdf = document?.file_name.toLowerCase().endsWith(".pdf");

  return (
    <section className="source-card">
      <div className="source-head">
        <span>{evidenceLabel(evidence)}</span>
        <strong>{document?.file_name || evidence.paper_name}</strong>
      </div>
      <p className="source-meta">
        {isPdf ? `第 ${evidence.page} 页` : `第 ${evidence.page} 段附近`}
        {evidence.section ? ` · ${evidence.section}` : ""}
      </p>
      <p className="evidence-caption">核心证据句</p>
      <mark>{coreEvidenceText(evidence)}</mark>
      <details className="source-context-details">
        <summary>展开上下文</summary>
        <p className="source-text">{maskSensitiveText(evidence.text)}</p>
      </details>
      <div className="source-actions">
        <a
          className="open-source"
          href={documentFileUrl(evidence.document_id, isPdf ? evidence.page : undefined)}
          target="_blank"
          rel="noreferrer"
        >
          {isPdf ? "打开原 PDF 位置" : "下载原 DOCX"}
        </a>
        {isPdf && (
          <button className="preview-source" type="button" onClick={() => setPreviewOpen((current) => !current)}>
            {previewOpen ? "收起右侧预览" : "在右侧预览 PDF"}
          </button>
        )}
      </div>
      {isPdf && !previewOpen && (
        <p className="pdf-preview-note">PDF 不会在刷新页面时自动加载；需要预览原文时再点击上方按钮。</p>
      )}
      {isPdf && previewOpen && (
        <iframe
          className="pdf-frame"
          title="PDF 原文预览"
          src={documentFileUrl(evidence.document_id, evidence.page)}
        />
      )}
    </section>
  );
}

function SettingsPanel({
  catalog,
  modelPreset,
  chatModel,
  embeddingModel,
  topK,
  chatModelOptions,
  embeddingModelOptions,
  onClose,
  onPresetChange,
  onChatModelChange,
  onEmbeddingModelChange,
  onTopKChange,
  onReset
}: {
  catalog: ModelCatalog | null;
  modelPreset: string;
  chatModel: string;
  embeddingModel: string;
  topK: number;
  chatModelOptions: string[];
  embeddingModelOptions: string[];
  onClose: () => void;
  onPresetChange: (value: string) => void;
  onChatModelChange: (value: string) => void;
  onEmbeddingModelChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onReset: () => void;
}) {
  const selectedPreset = catalog?.presets.find((item) => item.id === modelPreset);

  return (
    <div className="settings-backdrop" onClick={onClose}>
      <section
        className="settings-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="settings-head">
          <div>
            <span>基础配置</span>
            <h2 id="settings-title">模型和检索设置</h2>
            <p>这些配置会用于下一次提问。API Key 和服务地址仍然只放在后端环境变量里。</p>
          </div>
          <button className="settings-close" type="button" aria-label="关闭设置" onClick={onClose}>
            ×
          </button>
        </div>

        <label className="settings-field">
          <span>阅读模式</span>
          <select
            value={modelPreset}
            disabled={!catalog}
            onChange={(event) => onPresetChange(event.target.value)}
          >
            {(catalog?.presets || []).map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.label}
              </option>
            ))}
          </select>
          <small>{selectedPreset?.description || "后端模型目录加载中。"}</small>
        </label>

        <label className="settings-field">
          <span>对话模型</span>
          <select value={chatModel} onChange={(event) => onChatModelChange(event.target.value)} disabled={!chatModelOptions.length}>
            {chatModelOptions.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
          <small>负责理解问题、组织回答和引用证据。</small>
        </label>

        <label className="settings-field">
          <span>Embedding 模型</span>
          <select
            value={embeddingModel}
            onChange={(event) => onEmbeddingModelChange(event.target.value)}
            disabled={!embeddingModelOptions.length}
          >
            {embeddingModelOptions.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
          <small>负责把问题和文档片段变成向量，用于检索相似原文。</small>
        </label>

        <label className="settings-field">
          <span>检索数量 top-k</span>
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(event) => onTopKChange(Number(event.target.value))}
          />
          <small>数值越大，参考证据越多，但回答会更慢。建议普通问答 4-6，精读核对 7-10。</small>
        </label>

        <div className="settings-warning">
          更换 embedding 模型后，已有文档最好重新索引，否则“问题向量”和“文档向量”可能来自不同模型，检索质量会不稳定。
        </div>

        <div className="settings-actions">
          <button className="secondary-button" type="button" onClick={onReset} disabled={!catalog}>
            恢复默认
          </button>
          <button className="primary-button" type="button" onClick={onClose}>
            完成
          </button>
        </div>
      </section>
    </div>
  );
}

function intentLabel(value?: string) {
  const labels: Record<string, string> = {
    compound_request: "复合任务",
    reference_question: "参考文献问题",
    structured_review_request: "结构化阅读报告任务",
    compare_question: "多文档对比问题",
    title_alignment_question: "题目-结论匹配问题",
    reliability_question: "可靠性判断问题",
    research_limitation_question: "文章研究局限问题",
    document_wide_question: "整篇概括/分析问题",
    meta_question: "使用说明问题",
    specific_question: "具体内容问答"
  };
  return value ? labels[value] || value : "等待提问";
}

function retrievalStrategyLabel(value?: string) {
  const labels: Record<string, string> = {
    compound_request: "复合任务检索",
    reference_section: "参考文献区检索",
    structured_review: "结构化阅读报告检索",
    comparison_overview: "多文档概览检索",
    title_alignment: "题目与结论专项检索",
    reliability_check: "可靠性专项检索",
    research_limitation: "文章研究局限检索",
    document_overview: "整篇文档重点检索",
    vector_similarity: "向量相似度检索",
    hybrid_soft: "软意图混合检索",
    hybrid_reference: "参考文献混合检索",
    hybrid_field_lookup: "字段混合检索",
    hybrid_comparison: "对比混合检索",
    hybrid_judgment: "判断类混合检索",
    hybrid_limitation: "局限类混合检索",
    hybrid_overview: "全文概括混合检索",
    no_retrieval: "不检索文档"
  };
  return value ? labels[value] || value : "待运行";
}

function answerStrategyLabel(value?: string) {
  const labels: Record<string, string> = {
    local_compound_answer: "按用户顺序逐项回答复合任务",
    local_reference_answer: "本地规则整理参考文献",
    local_field_lookup_answer: "本地规则提取指定字段",
    local_structured_review_answer: "本地规则生成结构化阅读报告",
    local_title_alignment_answer: "本地规则判断题目匹配",
    local_reliability_answer: "本地规则判断可靠性",
    local_research_limitation_answer: "本地规则分析文章研究局限",
    local_compare_answer: "本地规则对比多文档",
    local_document_answer: "本地规则概括文档",
    missing_evidence_refusal: "证据不足，拒绝硬答",
    model_answer: "模型基于证据生成",
    model_unavailable: "模型不可用，未本地兜底",
    local_fallback_answer: "模型失败后本地降级回答"
  };
  return value ? labels[value] || value : "待生成";
}

function compoundTaskLabel(value: string) {
  const labels: Record<string, string> = {
    overview_summary: "总体概括",
    reference_list: "参考文献",
    professional_takeaways: "专业收获",
    reliability_judgment: "可靠性判断",
    method_analysis: "方法分析",
    limitation_analysis: "局限不足",
    conclusion_summary: "核心结论",
    comparison: "文档对比"
  };
  return labels[value] || value;
}

function evidenceQualityLabel(value?: string) {
  const labels: Record<string, string> = {
    strong: "证据充足",
    medium: "证据一般",
    weak: "证据偏弱",
    insufficient: "证据不足",
    fallback: "模型降级",
    none: "没有证据"
  };
  return value ? labels[value] || value : "待判断";
}

function TeachingPanel({
  documents,
  runtime,
  trace,
  latestQuestion,
  memoryUsed
}: {
  documents: DocumentInfo[];
  runtime: RuntimeStep[];
  trace: RagTrace | null;
  latestQuestion: string;
  memoryUsed: Record<string, string>;
}) {
  const readyDocumentCount = documents.filter((document) => document.status === "ready").length;
  const totalChunks = documents.reduce((sum, document) => sum + (document.chunk_count || 0), 0);
  const filteredDocumentCount = trace?.filter_document_ids?.length || 0;

  return (
    <section className="teaching-card">
      <div className="teaching-head">
        <span>开发者视图</span>
        <h2>本轮诊断</h2>
        <p>这里只展示后端本轮真实返回的 runtime、RAG trace 和证据信息，不再展示固定技术栈说明。</p>
      </div>

      <div className="question-card">
        <span>本轮问题</span>
        <strong>{latestQuestion || "还没有开始提问。上传文档并发送问题后，这里会显示本轮诊断。"}</strong>
      </div>

      <div className={`diagnosis-card ${trace?.evidence_quality || "pending"}`}>
        <span>{evidenceQualityLabel(trace?.evidence_quality)}</span>
        <strong>{trace?.diagnosis || "等待一次真实问答后显示诊断。"}</strong>
      </div>

      <div className="diagnostic-grid">
        <div>
          <span>问题类型</span>
          <strong>{intentLabel(trace?.intent)}</strong>
        </div>
        <div>
          <span>检索策略</span>
          <strong>{retrievalStrategyLabel(trace?.retrieval_strategy)}</strong>
        </div>
        <div>
          <span>回答策略</span>
          <strong>{answerStrategyLabel(trace?.answer_strategy)}</strong>
        </div>
        <div>
          <span>降级状态</span>
          <strong>{trace?.fallback_used ? "已降级" : trace ? "未降级" : "待运行"}</strong>
        </div>
      </div>

      {trace?.compound_tasks?.length ? (
        <div className="question-card">
          <span>子任务识别</span>
          <strong>{trace.compound_tasks.map(compoundTaskLabel).join(" → ")}</strong>
          {trace.task_parse_reason && <small>{trace.task_parse_reason}</small>}
        </div>
      ) : null}

      <details open className="teaching-section">
        <summary>真实 runtime</summary>
        <p>这些步骤来自后端本轮返回的 runtime，不额外添加固定流程。</p>
        <div className="pipeline-list">
          {runtime.length ? (
            runtime.map((step, index) => (
              <details key={`${step.node}-${step.title}-${index}`} open={index < 4}>
                <summary>
                  <span>{index + 1}</span>
                  <strong>{step.title}</strong>
                  <small>node：{step.node}</small>
                </summary>
                <p>{step.detail}</p>
              </details>
            ))
          ) : (
            <p>完成一次提问后，这里会显示后端实际经过的 runtime step。</p>
          )}
        </div>
      </details>

      <details open className="teaching-section">
        <summary>RAG 检索详情</summary>
        <p>这里保留开发者需要的 top-k、score、向量库记录数和 chunk 信息。</p>
        <div className="teaching-metrics">
          <div>
            <span>已准备文档</span>
            <strong>{readyDocumentCount}</strong>
          </div>
          <div>
            <span>文档切片</span>
            <strong>{totalChunks || "待生成"}</strong>
          </div>
          <div>
            <span>向量库</span>
            <strong>{trace?.vector_store || "Chroma"}</strong>
          </div>
          <div>
            <span>向量库记录</span>
            <strong>{trace?.vector_record_count ?? "待检索"}</strong>
          </div>
          <div>
            <span>本轮 top-k</span>
            <strong>{trace?.top_k || "待检索"}</strong>
          </div>
          <div>
            <span>返回证据</span>
            <strong>{trace?.retrieved_count ?? "待检索"}</strong>
          </div>
          <div>
            <span>筛选文档</span>
            <strong>{filteredDocumentCount || "全部/待定"}</strong>
          </div>
        </div>
        <div className="evidence-debug-list compact">
          {trace?.retrieval_debug?.length ? (
            trace.retrieval_debug.slice(0, 8).map((item) => (
              <article key={item.chunk_id}>
                <strong>
                  {item.citation_id} · chunk {item.chunk_id}
                </strong>
                <span>score：{formatScore(item.score)}</span>
                <span>page：{item.page}</span>
                <span>section：{item.section || "Unknown"}</span>
                <small className="hit-reason">命中原因：{item.reason}</small>
                <div className="debug-tags">
                  <span>{item.selected_by}</span>
                  <span>{item.used_in_answer ? "进入回答" : "未直接引用"}</span>
                  <span>{item.used_in_prompt ? "进入 prompt" : "未进 prompt"}</span>
                </div>
                {item.matched_keywords.length > 0 && (
                  <small>命中关键词：{item.matched_keywords.join("、")}</small>
                )}
                <details className="debug-evidence-details">
                  <summary>查看调试文本</summary>
                  <p>{maskSensitiveText(item.quote)}</p>
                </details>
              </article>
            ))
          ) : (
            <p>完成一次提问后，这里会显示本轮检索到的证据和 score。</p>
          )}
        </div>
      </details>

      <details className="teaching-section">
        <summary>文档索引与自动切分</summary>
        <p>上传后系统会自动解析文档、选择 chunk_size 和 overlap，并把每个片段写入向量数据库。</p>
        <div className="document-debug-list">
          {documents.length ? (
            documents.map((document) => (
              <article key={document.id}>
                <strong>{document.file_name}</strong>
                <span>状态：{humanDocumentStatus(document.status)}</span>
                <span>页数：{document.page_count || "待解析"}</span>
                <span>chunk 数：{document.chunk_count || "待生成"}</span>
                <span>embedding：{friendlyEmbeddingLabel(document.embedding_model)}</span>
                {document.chunk_strategy ? (
                  <>
                    <span>chunk_size：{document.chunk_strategy.chunk_size}</span>
                    <span>overlap：{document.chunk_strategy.overlap}</span>
                    <span>splitter：{document.chunk_strategy.splitter}</span>
                    <span>语言：{document.chunk_strategy.language}</span>
                    <small>切分理由：{document.chunk_strategy.reasons.join("；")}</small>
                  </>
                ) : (
                  <small>{document.error || "文档准备完成后会显示自动切分理由。"}</small>
                )}
              </article>
            ))
          ) : (
            <p>还没有上传文档。</p>
          )}
        </div>
      </details>

      <details className="teaching-section">
        <summary>生成回答时使用了什么</summary>
        <p>这里展示最终回答前放进 Prompt 的证据摘要。界面会隐藏 API 地址、密钥和原始模型 ID。</p>
        <div className="trace-table">
          <span>阅读方式</span>
          <strong>{trace?.model_profile || "待运行"}</strong>
          <span>问题类型</span>
          <strong>{intentLabel(trace?.intent)}</strong>
          <span>回答策略</span>
          <strong>{answerStrategyLabel(trace?.answer_strategy)}</strong>
        </div>
        <div className="final-evidence-list">
          {trace?.final_prompt_evidence.length ? (
            trace.final_prompt_evidence.map((item) => <small key={item}>{item}</small>)
          ) : (
            <p>完成一次提问后，这里会显示最终 prompt 使用了哪些证据。</p>
          )}
        </div>
      </details>

      <details className="teaching-section">
        <summary>记忆系统</summary>
        <p>系统会区分短期对话和长期偏好，例如用户背景、解释习惯、当前论文上下文。</p>
        {Object.keys(memoryUsed).length ? (
          <div className="memory-list">
            {Object.entries(memoryUsed).map(([key, value]) => (
              <span key={key}>
                <strong>{memoryLabel(key)}</strong>
                {value}
              </span>
            ))}
          </div>
        ) : (
          <p>本轮还没有可展示的长期画像或偏好。</p>
        )}
      </details>
    </section>
  );
}

function renderAnswerWithCitations(
  answer: string,
  evidence: EvidenceItem[],
  onPick: (item: EvidenceItem) => void
) {
  const normalized = normalizeAnswerMarkdown(answer);
  const blocks = parseMarkdownBlocks(normalized);
  return (
    <div className="answer-markdown">
      {blocks.map((block, index) => renderMarkdownBlock(block, index, evidence, onPick))}
    </div>
  );
}

type MarkdownBlock =
  | { type: "heading"; level: 2 | 3; text: string }
  | { type: "paragraph"; text: string }
  | { type: "list"; ordered: boolean; items: string[] };

function normalizeAnswerMarkdown(answer: string): string {
  return answer
    .replace(/\r\n/g, "\n")
    .replace(/\[(?:Introduction|Conclusion|Abstract|Methods?|Results?|Discussion|Unknown)依据\]/gi, "")
    .replace(/(?:Introduction|Conclusion|Abstract|Methods?|Results?|Discussion|Unknown)依据/gi, "")
    .replace(/([^\n])\n(?=(?:[-*]\s+|\d+[.、]\s+|#{1,3}\s+))/g, "$1\n\n")
    .trim();
}

function parseMarkdownBlocks(markdown: string): MarkdownBlock[] {
  const lines = markdown.split("\n");
  const blocks: MarkdownBlock[] = [];
  let paragraph: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  function flushParagraph() {
    const text = paragraph.join(" ").replace(/\s+/g, " ").trim();
    if (text) {
      blocks.push({ type: "paragraph", text });
    }
    paragraph = [];
  }

  function flushList() {
    if (list?.items.length) {
      blocks.push({ type: "list", ordered: list.ordered, items: list.items });
    }
    list = null;
  }

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ type: "heading", level: heading[1].length === 1 ? 2 : 3, text: heading[2].trim() });
      continue;
    }

    const unorderedItem = line.match(/^[-*]\s+(.+)$/);
    const orderedItem = line.match(/^\d+[.、]\s+(.+)$/);
    if (unorderedItem || orderedItem) {
      flushParagraph();
      const ordered = Boolean(orderedItem);
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push((unorderedItem?.[1] || orderedItem?.[1] || "").trim());
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();
  return blocks;
}

function renderMarkdownBlock(
  block: MarkdownBlock,
  index: number,
  evidence: EvidenceItem[],
  onPick: (item: EvidenceItem) => void
) {
  if (block.type === "heading") {
    const HeadingTag = block.level === 2 ? "h3" : "h4";
    return <HeadingTag key={`heading-${index}`}>{renderInlineMarkdown(block.text, evidence, onPick)}</HeadingTag>;
  }
  if (block.type === "list") {
    const ListTag = block.ordered ? "ol" : "ul";
    return (
      <ListTag key={`list-${index}`}>
        {block.items.map((item, itemIndex) => (
          <li key={`item-${index}-${itemIndex}`}>{renderInlineMarkdown(item, evidence, onPick)}</li>
        ))}
      </ListTag>
    );
  }
  return <p key={`paragraph-${index}`}>{renderInlineMarkdown(block.text, evidence, onPick)}</p>;
}

function renderInlineMarkdown(
  text: string,
  evidence: EvidenceItem[],
  onPick: (item: EvidenceItem) => void
): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|\[E\d+\])/g).filter(Boolean);
  return parts.map((part, index) => {
    const citation = part.match(/^\[E(\d+)\]$/);
    if (citation) {
      const item = evidence.find((candidate) => candidate.citation_id === `E${citation[1]}`);
      if (!item) {
        return <Fragment key={`${part}-${index}`}>{part}</Fragment>;
      }
      return (
        <button className="inline-citation" key={`${part}-${index}`} onClick={() => onPick(item)}>
          {evidenceLabel(item)}
        </button>
      );
    }
    const bold = part.match(/^\*\*(.+)\*\*$/);
    if (bold) {
      return <strong key={`${part}-${index}`}>{renderInlineMarkdown(bold[1], evidence, onPick)}</strong>;
    }
    return <Fragment key={`${part}-${index}`}>{part}</Fragment>;
  });
}

function renderEvidenceActions(message: ChatMessage, onPick: (item: EvidenceItem) => void) {
  const evidence = message.evidence || [];
  if (!evidence.length) {
    return null;
  }
  const inlineCitationIds = citationIdsFromAnswer(message.content);
  const primaryEvidence = inlineCitationIds.length ? [] : primaryEvidenceForAnswer(message.content, evidence);
  const moreEvidence: EvidenceItem[] = [];

  if (!primaryEvidence.length && !moreEvidence.length) {
    return null;
  }

  return (
    <div className="citation-row">
      {primaryEvidence.map((item) => (
        <button key={item.chunk_id} onClick={() => onPick(item)}>
          {evidenceLabel(item)}
        </button>
      ))}
      {moreEvidence.length > 0 && (
        <details className="citation-more">
          <summary>更多证据（{moreEvidence.length}）</summary>
          <div>
            {moreEvidence.map((item) => (
              <button key={item.chunk_id} onClick={() => onPick(item)}>
                {evidenceLabel(item)} · {item.citation_id}
              </button>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function citationIdsFromAnswer(answer: string): string[] {
  return Array.from(new Set(Array.from(answer.matchAll(/\[E(\d+)\]/g), (match) => `E${match[1]}`)));
}

function primaryEvidenceForAnswer(answer: string, evidence: EvidenceItem[]): EvidenceItem[] {
  const citedIds = citationIdsFromAnswer(answer);
  return citedIds
    .map((citationId) => evidence.find((item) => item.citation_id === citationId))
    .filter((item): item is EvidenceItem => Boolean(item));
}

function evidenceLabel(item: EvidenceItem): string {
  return `证据 ${item.citation_id.replace("E", "")}`;
}

function humanDocumentStatus(status: string): string {
  const labels: Record<string, string> = {
    queued: "等待准备",
    processing: "准备中",
    ready: "可提问",
    failed: "需要重试"
  };
  return labels[status] || status;
}

function formatScore(score: number): string {
  if (!Number.isFinite(score)) {
    return "未知";
  }
  return score.toFixed(3);
}

function formatPercent(value: number): string {
  if (!Number.isFinite(value)) {
    return "--";
  }
  return `${Math.round(value * 100)}%`;
}

function formatLatency(milliseconds: number): string {
  if (!Number.isFinite(milliseconds)) {
    return "--";
  }
  if (milliseconds >= 1000) {
    return `${(milliseconds / 1000).toFixed(1)} 秒`;
  }
  return `${milliseconds} 毫秒`;
}

function coreEvidenceText(evidence: EvidenceItem): string {
  const raw = maskSensitiveText(evidence.quote || evidence.text);
  const candidates = raw
    .split(/\n+|(?<=[。！？.!?；;])\s*/g)
    .map((part) => part.split(/\s+/).join(" ").trim())
    .filter(Boolean);
  const readable = candidates.find((part) => part.length >= 12 && !looksLikeTableEvidence(part));
  const normalized = readable || raw.split(/\s+/).join(" ");

  if (!readable && looksLikeTableEvidence(normalized)) {
    return "这条证据主要是表格或公式，已折叠。需要核对数值时可以展开上下文查看。";
  }
  if (normalized.length <= 140) {
    return normalized;
  }
  const sentenceEnd = normalized.search(/[。！？.!?；;]/);
  if (sentenceEnd > 40 && sentenceEnd <= 140) {
    return normalized.slice(0, sentenceEnd + 1);
  }
  return `${normalized.slice(0, 140)}...`;
}

function looksLikeTableEvidence(text: string): boolean {
  const normalized = text.split(/\s+/).join(" ");
  const pipeCount = (normalized.match(/\|/g) || []).length;
  const digitCount = (normalized.match(/\d/g) || []).length;
  const cjkCount = (normalized.match(/[\u4e00-\u9fff]/g) || []).length;
  const separatorCount = (normalized.match(/[|,:：/\\]/g) || []).length;
  return (
    pipeCount >= 6 ||
    (normalized.length > 100 && digitCount / normalized.length > 0.28 && cjkCount < 80) ||
    (normalized.length > 120 && separatorCount / normalized.length > 0.22)
  );
}

function friendlyEmbeddingLabel(model?: string | null): string {
  if (!model) {
    return "待生成";
  }
  if (model === "本地备用检索") {
    return model;
  }
  return "已配置的 embedding 模型";
}

function memoryLabel(key: string): string {
  const labels: Record<string, string> = {
    user_profile: "用户画像",
    explanation_style: "解释风格",
    preference: "长期偏好"
  };
  return labels[key] || key;
}

function maskSensitiveText(text: string): string {
  return text
    .replace(/[\w.+-]+@[\w.-]+\.\w+/g, "[邮箱已隐藏]")
    .replace(/(姓\s*名|姓名)\s*[:：]?\s*[^\s，。；;|]+/g, "$1：[已隐藏]")
    .replace(/(学\s*号|学号)\s*[:：]?\s*[^\s，。；;|]+/g, "$1：[已隐藏]")
    .replace(/(电子邮件|邮箱)\s*[:：]?\s*[^\s，。；;|]+/g, "$1：[已隐藏]");
}

function toFriendlyError(message: string): string {
  if (message.includes("Failed to fetch")) {
    return "暂时连接不上后端服务。请确认后端窗口还在运行。";
  }
  if (message.includes("模型") || message.toLowerCase().includes("api")) {
    return message;
  }
  return message || "操作失败，请稍后再试。";
}

export default App;
