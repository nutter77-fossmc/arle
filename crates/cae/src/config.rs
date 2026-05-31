use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum tokens per generation.
    pub max_tokens: usize,
    /// Temperature for sampling.
    pub temperature: f32,
    /// Top-p for nucleus sampling.
    pub top_p: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaeConfig {
    /// Path to the aggregator model (Qwen3.5-2B).
    pub aggregator_model_path: String,

    /// Path to the base expert model (Qwen3.5-0.8B).
    pub base_expert_model_path: String,

    /// Path to the shared baseline LoRA adapter.
    pub baseline_lora_path: Option<String>,

    /// Directory containing all 16 domain LoRA adapters.
    pub adapters_dir: Option<String>,

    /// Pipeline steps to execute.
    pub pipeline_steps: Vec<PipelineStep>,

    /// Maximum context length (tokens).
    pub max_context_length: usize,

    /// KV cache quantization (4-bit or bf16).
    pub kv_cache_4bit: bool,

    /// Device backend (auto, metal, cpu).
    pub backend: String,

    /// Whether to save conversation memory to iCloud.
    pub memory_enabled: bool,

    /// iCloud memory directory path.
    pub memory_path: Option<String>,

    /// Inference generation parameters.
    pub inference: InferenceConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStep {
    Plan,
    Draft,
    Review,
    Revise,
    Assess,
    Submit,
}

impl CaeConfig {
    pub fn default_m4() -> Self {
        Self {
            aggregator_model_path: "Qwen/Qwen3.5-2B".into(),
            base_expert_model_path: "Qwen/Qwen3.5-0.8B".into(),
            baseline_lora_path: None,
            adapters_dir: None,
            pipeline_steps: vec![
                PipelineStep::Plan,
                PipelineStep::Draft,
                PipelineStep::Review,
                PipelineStep::Revise,
                PipelineStep::Assess,
                PipelineStep::Submit,
            ],
            max_context_length: 262_144,
            kv_cache_4bit: false,
            backend: "metal".into(),
            memory_enabled: true,
            memory_path: Some(
                "~/Library/Mobile Documents/com~apple~CloudDocs/agent-memory/".into(),
            ),
            inference: InferenceConfig {
                max_tokens: 512,
                temperature: 0.7,
                top_p: 0.9,
            },
        }
    }

    pub fn default_m1() -> Self {
        Self {
            kv_cache_4bit: true,
            inference: InferenceConfig {
                max_tokens: 256,
                temperature: 0.7,
                top_p: 0.9,
            },
            ..Self::default_m4()
        }
    }
}
