use crate::registry::CaeExpert;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterState {
    Unloaded,
    Loading,
    Active,
    Unloading,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterInfo {
    pub expert_id: usize,
    pub expert_name: String,
    pub adapter_path: Option<String>,
    pub baseline_merged: bool,
    pub state: AdapterState,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct AdapterManager {
    /// Currently active expert.
    active_expert: Option<CaeExpert>,
    /// Path to the base Qwen3.5-0.8B model.
    base_model_path: String,
    /// Path to the baseline LoRA adapter.
    baseline_lora_path: Option<String>,
    /// Directory containing domain LoRA adapters.
    adapters_dir: Option<String>,
    /// Adapter states for tracking.
    adapter_states: Vec<AdapterInfo>,
}

impl AdapterManager {
    pub fn new(
        base_model_path: impl Into<String>,
        baseline_lora_path: Option<String>,
        adapters_dir: Option<String>,
    ) -> Self {
        Self {
            active_expert: None,
            base_model_path: base_model_path.into(),
            baseline_lora_path,
            adapters_dir,
            adapter_states: Vec::new(),
        }
    }

    pub fn load_adapter(&mut self, expert: &CaeExpert) -> Result<()> {
        let adapter_path = self
            .adapters_dir
            .as_ref()
            .map(|dir| format!("{}/{}", dir, expert.adapter_file));

        self.adapter_states.push(AdapterInfo {
            expert_id: expert.id,
            expert_name: expert.name.clone(),
            adapter_path: adapter_path.clone(),
            baseline_merged: true,
            state: AdapterState::Active,
        });

        self.active_expert = Some(expert.clone());
        Ok(())
    }

    pub fn unload_adapter(&mut self, expert_id: usize) -> Result<()> {
        if let Some(entry) = self
            .adapter_states
            .iter_mut()
            .find(|a| a.expert_id == expert_id)
        {
            entry.state = AdapterState::Unloaded;
            self.active_expert = None;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Expert {} not found", expert_id))
        }
    }

    pub fn swap_adapter(&mut self, new_expert: &CaeExpert) -> Result<()> {
        if let Some(current) = &self.active_expert {
            if current.id == new_expert.id {
                return Ok(()); // already loaded
            }
            self.unload_adapter(current.id)?;
        }
        self.load_adapter(new_expert)?;
        Ok(())
    }

    pub fn active_expert(&self) -> Option<&CaeExpert> {
        self.active_expert.as_ref()
    }

    pub fn is_loaded(&self, expert_id: usize) -> bool {
        self.adapter_states
            .iter()
            .any(|a| a.expert_id == expert_id && a.state == AdapterState::Active)
    }

    pub fn state_summary(&self) -> Vec<AdapterInfo> {
        self.adapter_states.clone()
    }

    /// Memory estimate for loaded adapters in MB.
    pub fn memory_estimate_mb(&self) -> f32 {
        // Base 0.8B model: ~1.6 GB
        // Each LoRA: ~5 MB
        let base_mb = 1.6 * 1024.0;
        let lora_mb = 5.0 * self.adapter_states.iter().filter(|a| {
            matches!(a.state, AdapterState::Active | AdapterState::Loading)
        }).count() as f32;
        base_mb + lora_mb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::CaeRegistry;

    #[test]
    fn adapter_load_then_unload() {
        let registry = CaeRegistry::new();
        let math = registry.get(1).unwrap();
        let mut mgr = AdapterManager::new("dummy/path", None, None);

        mgr.load_adapter(math).unwrap();
        assert!(mgr.is_loaded(1));
        assert_eq!(mgr.active_expert().unwrap().name, "math");

        mgr.unload_adapter(1).unwrap();
        assert!(!mgr.is_loaded(1));
        assert!(mgr.active_expert().is_none());
    }

    #[test]
    fn adapter_swap_changes_expert() {
        let registry = CaeRegistry::new();
        let math = registry.get(1).unwrap();
        let code = registry.get(2).unwrap();
        let mut mgr = AdapterManager::new("dummy/path", None, None);

        mgr.load_adapter(math).unwrap();
        assert_eq!(mgr.active_expert().unwrap().id, 1);

        mgr.swap_adapter(code).unwrap();
        assert_eq!(mgr.active_expert().unwrap().id, 2);
    }

    #[test]
    fn swap_to_same_expert_is_noop() {
        let registry = CaeRegistry::new();
        let math = registry.get(1).unwrap();
        let mut mgr = AdapterManager::new("dummy/path", None, None);

        mgr.load_adapter(math).unwrap();
        mgr.swap_adapter(math).unwrap(); // same expert
        assert!(mgr.is_loaded(1));
    }
}
