# PDF Corpus For Real Evaluation

This folder contains public PDFs prepared for the mixed PDF/DOCX evaluation suite.

| File | Source URL | Evaluation Purpose |
|---|---|---|
| `nist_ai_rmf_1_0.pdf` | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf | Official framework, text, tables, lifecycle figures |
| `nist_cybersecurity_framework_2_0.pdf` | https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf | Official framework, functions, profiles, governance |
| `attention_is_all_you_need.pdf` | https://arxiv.org/pdf/1706.03762.pdf | Research paper, architecture figure, BLEU table |
| `bert_pretraining_bidirectional_transformers.pdf` | https://arxiv.org/pdf/1810.04805.pdf | Research paper, pretraining objectives, benchmark tables |
| `clip_learning_transferable_visual_models.pdf` | https://proceedings.mlr.press/v139/radford21a/radford21a.pdf | Vision-language paper, zero-shot transfer, contrastive learning |
| `gpt2_language_models_unsupervised_multitask_learners.pdf` | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | Language model paper, WebText, zero-shot tasks |
| `segment_anything.pdf` | https://arxiv.org/pdf/2304.02643.pdf | Image-heavy paper, promptable segmentation, SA-1B |
| `ocrmypdf_cardinal_scanned.pdf` | https://raw.githubusercontent.com/ocrmypdf/OCRmyPDF/main/tests/resources/cardinal.pdf | Scanned/image-only OCR regression PDF |

Current indexing notes:

- All eight files have been indexed as ready documents.
- Vision analysis was enabled with a per-document cap during indexing.
- The scanned OCR test PDF has no extractable text layer; current usable evidence comes from image/vision summaries rather than `ocr_text`.
