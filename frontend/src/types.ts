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
  token_count?: number;
  size_unit?: string;
  parent_chunk_size?: number;
  parent_overlap?: number;
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

export interface RelatedImageInfo {
  id: string;
  document_id: string;
  page_start: number;
  page_end: number;
  kind: string;
  caption_text?: string;
  ocr_text?: string;
  ocr_status?: string;
  ocr_error?: string;
  vision_summary?: string;
  vision_error?: string;
  status?: string;
}

export interface EvidenceItem {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  paper_name: string;
  page: number;
  page_start?: number | null;
  page_end?: number | null;
  section?: string | null;
  source: string;
  file_hash: string;
  score: number;
  vector_score?: number | null;
  sparse_score?: number | null;
  rule_score?: number | null;
  rrf_score?: number | null;
  final_score?: number | null;
  score_source?: string;
  text: string;
  quote: string;
  char_start?: number | null;
  char_end?: number | null;
  token_count?: number | null;
  chunk_type?: string;
  parent_id?: string | null;
  image_id?: string | null;
  image_path?: string | null;
  bbox_json?: string | null;
  related_images?: RelatedImageInfo[];
  quality_label?: string;
  quality_reasons?: string[];
  selection_status?: string;
  rejection_reason?: string;
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
  page_start?: number | null;
  page_end?: number | null;
  section?: string | null;
  score: number;
  vector_score?: number | null;
  sparse_score?: number | null;
  rule_score?: number | null;
  rrf_score?: number | null;
  final_score?: number | null;
  score_source?: string;
  retrieval_strategy: string;
  selected_by: string;
  matched_keywords: string[];
  reason: string;
  used_in_answer: boolean;
  used_in_prompt: boolean;
  quote: string;
  quality_label?: string;
  quality_reasons?: string[];
  selection_status?: string;
  rejection_reason?: string;
}

export interface EvidenceQualityTraceItem {
  citation_id?: string;
  chunk_id: string;
  document_id: string;
  paper_name?: string;
  page: number;
  page_start?: number | null;
  page_end?: number | null;
  section?: string | null;
  chunk_type?: string;
  candidate_rank: number;
  selected_rank?: number | null;
  selection_status: string;
  quality_label: string;
  quality_reasons: string[];
  rejection_reason?: string;
  score: number;
  relevance_score: number;
  readability_score: number;
  vector_score?: number | null;
  sparse_score?: number | null;
  rule_score?: number | null;
  rrf_score?: number | null;
  final_score?: number | null;
  score_source?: string;
  matched_keywords?: string[];
  judge_verdict?: string;
  judge_reason?: string;
  judge_confidence?: number | null;
  quote?: string;
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

export interface VisualOcrWarning {
  type: string;
  severity: "info" | "warn" | string;
  document_id?: string;
  paper_name?: string;
  image_count?: number;
  vision_ready_count?: number;
  unfinished_count?: number;
  ocr_failed_count?: number;
  ocr_empty_count?: number;
  ocr_skipped_count?: number;
  status_counts?: Record<string, number>;
  ocr_status_counts?: Record<string, number>;
  message: string;
}

export interface MultiDocumentCard {
  document_id: string;
  paper_name: string;
  covered: boolean;
  evidence_count: number;
  citation_ids: string[];
  pages: string[];
  key_terms: string[];
  roles: Array<{
    role: string;
    label: string;
    score: number;
  }>;
  best_quote?: string;
  evidence_types?: string[];
  image_evidence_count?: number;
}

export interface DocumentRelationItem {
  source_document_id: string;
  target_document_id: string;
  source_name: string;
  target_name: string;
  relation_type: string;
  relation_label: string;
  shared_terms: string[];
  source_citations: string[];
  target_citations: string[];
  summary: string;
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
  retrieval_pipeline?: string;
  ranking_method?: string;
  embedding_requested_model?: string;
  embedding_provider?: string;
  embedding_used_fallback?: boolean;
  embedding_fallback_reason?: string;
  embedding_document_fallback_count?: number;
  embedding_document_providers?: Record<string, string>;
  answer_strategy?: string;
  fallback_used?: boolean;
  evidence_quality?: string;
  evidence_coverage?: {
    should_refuse?: boolean;
    reason_code?: string;
    reason?: string;
    metrics?: Record<string, unknown>;
  };
  diagnosis?: string;
  retrieval_debug?: RetrievalDebugItem[];
  evidence_quality_trace?: EvidenceQualityTraceItem[];
  compound_tasks?: string[];
  task_parse_reason?: string;
  evidence_judgments?: EvidenceJudgment[];
  verification?: VerificationTrace;
  multi_document_cards?: MultiDocumentCard[];
  document_relation_map?: DocumentRelationItem[];
  multi_document_coverage?: Record<string, unknown>;
  visual_ocr_warnings?: VisualOcrWarning[];
  agent_calls?: Array<Record<string, unknown>>;
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
  chat_model: string;
  embedding_model: string;
  top_k: number;
}

export interface ModelCatalog {
  presets: ModelPreset[];
  default_preset: string;
  chat_model_options: string[];
  embedding_model_options: string[];
  default_chat_model: string;
  default_embedding_model: string;
  default_top_k: number;
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
