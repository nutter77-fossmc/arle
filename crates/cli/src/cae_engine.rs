use anyhow::Result;
use cae::InferenceProvider;
use infer::backend::metal::MetalBackend;
use infer::backend::InferenceBackend;
use infer::sampler::SamplingParams;
use std::path::Path;

pub struct MetalCaeEngine {
    backend: MetalBackend,
    _model_path: String,
}

impl MetalCaeEngine {
    pub fn new(model_path: impl Into<String>) -> Result<Self> {
        let path = model_path.into();
        let mut backend = MetalBackend::new();
        backend.load(Path::new(&path))?;
        Ok(Self {
            backend,
            _model_path: path,
        })
    }
}

impl InferenceProvider for MetalCaeEngine {
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let params = SamplingParams {
            max_new_tokens: Some(max_tokens),
            temperature: 0.7,
            top_k: -1,
            top_p: 0.9,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            ignore_eos: false,
            stop_token_ids: vec![],
            seed: None,
        };
        let result = self.backend.generate(prompt, &params)?;
        Ok(result.text)
    }

    fn load_expert(&mut self, _expert_id: usize) -> Result<()> {
        anyhow::bail!("adapter hot-swap not yet implemented (Phase C)")
    }

    fn unload_expert(&mut self) -> Result<()> {
        anyhow::bail!("adapter hot-swap not yet implemented (Phase C)")
    }

    fn name(&self) -> &str {
        "metal"
    }

    fn supports_adapter_swap(&self) -> bool {
        false
    }
}
