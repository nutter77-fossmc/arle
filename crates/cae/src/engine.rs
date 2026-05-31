use anyhow::Result;

/// Provider trait for model inference in the CAE pipeline.
/// Backend-agnostic — implemented by Metal, CUDA, or mock backends.
pub trait InferenceProvider: Send {
    /// Run a completion with the given prompt and return generated text.
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String>;

    /// Load a specific expert's LoRA adapter by ID (1-16).
    fn load_expert(&mut self, expert_id: usize) -> Result<()>;

    /// Unload the current expert adapter, returning to base model.
    fn unload_expert(&mut self) -> Result<()>;

    /// Get a descriptive name for this provider.
    fn name(&self) -> &str;

    /// Whether the provider supports adapter hot-swap.
    fn supports_adapter_swap(&self) -> bool {
        false
    }
}
