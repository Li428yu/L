export type DocumentStatus = "queued" | "processing" | "ready" | "failed";
export type TaskStatus = "queued" | "running" | "completed" | "failed";

export interface ChunkStrategy {
  chunk_size: number;
  overlap: number;
  splitter: string;
  language: string;
  page_count: number;
  paragraph_count: number;
  char_count: number;
  reasons: string[];
}

export interface DocumentInfo {
  id: string;
  file_name: string;
  file_hash: string;
  status: DocumentStatus;
  page_count: number;
  chunk_count: number;
  source_path: string;
  embedding_model?: string | null;
  chunk_strategy?: ChunkStrategy | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface TaskInfo {
  id: string;
  document_id?: string | null;
  stage: string;
  status: TaskStatus;
  progress: number;
  message: string;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface UploadResponse {
  document: DocumentInfo;
  task: TaskInfo;
}

export interface EvidenceItem {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  paper_name: string;
  page: number;
  section?: string | null;
  source: string;
  file_hash: string;
  score: number;
  text: string;
  quote: string;
  char_start?: number | null;
  char_end?: number | null;
}

export interface RuntimeStep {
  node: string;
  title: string;
  detail: string;
}

export interface RetrievalDebugItem {
  citation_id: string;
  chunk_id: string;
  page: number;
  section?: string | null;
  score: number;
  retrieval_strategy: string;
  selected_by: string;
  matched_keywords: string[];
  reason: string;
  used_in_answer: boolean;
  used_in_prompt: boolean;
  quote: string;
}

export interface EvidenceJudgment {
  citation_id: string;
  chunk_id: string;
  verdict: string;
  confidence: number;
  reason: string;
  retrieval_strategy: string;
}

export interface VerificationTrace {
  status?: string;
  summary?: string;
  citation_count?: number;
  missing_citations?: string[];
  weak_citations?: Array<{
    citation_id: string;
    overlap: number;
    reason: string;
  }>;
  uncited_answer?: boolean;
}

export interface RagTrace {
  model_profile: string;
  vector_store: string;
  vector_record_count: number;
  top_k: number;
  filter_document_ids: string[];
  retrieved_count: number;
  final_prompt_evidence: string[];
  intent?: string;
  retrieval_strategy?: string;
  answer_strategy?: string;
  fallback_used?: boolean;
  evidence_quality?: string;
  diagnosis?: string;
  retrieval_debug?: RetrievalDebugItem[];
  compound_tasks?: string[];
  task_parse_reason?: string;
  evidence_judgments?: EvidenceJudgment[];
  verification?: VerificationTrace;
}

export interface AskResponse {
  answer: string;
  conversation_id: string;
  evidence: EvidenceItem[];
  runtime: RuntimeStep[];
  rag_trace: RagTrace;
  memory_used: Record<string, string>;
}

export interface ConversationInfo {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessage {
  id: number;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  evidence: EvidenceItem[];
  created_at: string;
}

export interface ConversationDetail {
  conversation: ConversationInfo;
  messages: ConversationMessage[];
  memory_used: Record<string, string>;
}

export interface ModelPreset {
  id: string;
  label: string;
  description: string;
}

export interface ModelCatalog {
  presets: ModelPreset[];
  default_preset: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  evidence?: EvidenceItem[];
  runtime?: RuntimeStep[];
  rag_trace?: RagTrace;
  memory_used?: Record<string, string>;
}
