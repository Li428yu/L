import type {
  AskResponse,
  ConversationDetail,
  ConversationInfo,
  DocumentInfo,
  ModelCatalog,
  TaskInfo,
  UploadResponse
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `请求失败：${response.status}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function listDocuments(): Promise<DocumentInfo[]> {
  return request<DocumentInfo[]>("/api/documents");
}

export function listConversations(): Promise<ConversationInfo[]> {
  return request<ConversationInfo[]>("/api/conversations");
}

export function getConversation(conversationId: string): Promise<ConversationDetail> {
  return request<ConversationDetail>(`/api/conversations/${conversationId}`);
}

export function deleteConversation(conversationId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/conversations/${conversationId}`, {
    method: "DELETE"
  });
}

export function getModels(): Promise<ModelCatalog> {
  return request<ModelCatalog>("/api/models");
}

export function uploadDocument(file: File): Promise<UploadResponse> {
  const body = new FormData();
  body.append("file", file);
  return request<UploadResponse>("/api/documents/upload", {
    method: "POST",
    body
  });
}

export function getTask(taskId: string): Promise<TaskInfo> {
  return request<TaskInfo>(`/api/tasks/${taskId}`);
}

export function deleteDocument(documentId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/documents/${documentId}`, {
    method: "DELETE"
  });
}

export async function askPaperStream(
  payload: {
    question: string;
    conversation_id?: string | null;
    document_ids: string[];
    model_preset: string;
    chat_model?: string;
    embedding_model?: string;
    top_k?: number;
  },
  handlers: {
    onChunk: (chunk: string) => void;
    onStatus?: (status: string) => void;
  }
): Promise<AskResponse> {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok || !response.body) {
    let message = `请求失败：${response.status}`;
    try {
      const errorPayload = await response.json();
      message = errorPayload.detail || message;
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload: AskResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      const event = JSON.parse(line) as { type: string; payload: unknown };
      if (event.type === "status" && typeof event.payload === "string") {
        handlers.onStatus?.(event.payload);
      } else if ((event.type === "token" || event.type === "chunk") && typeof event.payload === "string") {
        handlers.onChunk(event.payload);
      } else if (event.type === "final") {
        finalPayload = event.payload as AskResponse;
      } else if (event.type === "error") {
        throw new Error(String(event.payload || "回答生成失败"));
      }
    }
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer) as { type: string; payload: unknown };
    if (event.type === "final") {
      finalPayload = event.payload as AskResponse;
    }
  }
  if (!finalPayload) {
    throw new Error("回答没有完整返回，请重试。");
  }
  return finalPayload;
}

export function documentFileUrl(documentId: string, page?: number): string {
  if (!page) {
    return `${API_BASE}/api/documents/${documentId}/file`;
  }
  return `${API_BASE}/api/documents/${documentId}/file#page=${page}`;
}

export function evidenceImageUrl(documentId: string, imageId: string, thumbnail = false): string {
  const suffix = thumbnail ? "?thumbnail=true" : "";
  return `${API_BASE}/api/documents/${documentId}/images/${encodeURIComponent(imageId)}/file${suffix}`;
}
