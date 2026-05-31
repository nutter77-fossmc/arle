use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaeExpert {
    /// Expert ID (1-16).
    pub id: usize,
    /// Short name (e.g. "math", "code").
    pub name: String,
    /// Full domain name.
    pub domain: String,
    /// Description of expertise.
    pub description: String,
    /// Domain-specific adapter filename (without path).
    pub adapter_file: String,
    /// HF collection tags for training data.
    pub collection_tags: Vec<String>,
    /// Whether this expert can participate in review phase.
    pub can_review: bool,
    /// Whether this expert can draft independently.
    pub can_draft: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaeRegistry {
    pub experts: Vec<CaeExpert>,
}

impl CaeRegistry {
    /// Create the full 16-expert registry matching the CAE plan.
    pub fn new() -> Self {
        let experts = vec![
            CaeExpert {
                id: 1,
                name: "math".into(),
                domain: "Mathematics".into(),
                description: "Math reasoning, math QA, MCQA".into(),
                adapter_file: "cae_01_math.safetensors".into(),
                collection_tags: vec!["math-reasoning".into(), "math-qa".into(), "mcqa".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 2,
                name: "code".into(),
                domain: "Software Engineering".into(),
                description: "General code generation and software engineering".into(),
                adapter_file: "cae_02_code.safetensors".into(),
                collection_tags: vec![
                    "code".into(),
                    "code-execution".into(),
                    "code-plus".into(),
                ],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 3,
                name: "rust".into(),
                domain: "Rust Programming".into(),
                description: "Rust-specific coding and systems programming".into(),
                adapter_file: "cae_03_rust.safetensors".into(),
                collection_tags: vec!["rust".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 4,
                name: "swift".into(),
                domain: "Swift/iOS".into(),
                description: "Swift and iOS development".into(),
                adapter_file: "cae_04_swift.safetensors".into(),
                collection_tags: vec!["swift".into(), "ios".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 5,
                name: "physics".into(),
                domain: "Physics".into(),
                description: "Scientific physics and physical reasoning".into(),
                adapter_file: "cae_05_physics.safetensors".into(),
                collection_tags: vec!["science-physics".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 6,
                name: "biology".into(),
                domain: "Life Sciences".into(),
                description: "Biology and life sciences".into(),
                adapter_file: "cae_06_biology.safetensors".into(),
                collection_tags: vec!["biology".into(), "life-sciences".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 7,
                name: "chemistry".into(),
                domain: "Chemistry".into(),
                description: "Chemistry and chemical reasoning".into(),
                adapter_file: "cae_07_chemistry.safetensors".into(),
                collection_tags: vec!["chemistry".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 8,
                name: "creative-writing".into(),
                domain: "Creative Writing".into(),
                description: "Creative prose, fiction, storytelling".into(),
                adapter_file: "cae_08_creative_writing.safetensors".into(),
                collection_tags: vec![
                    "write".into(),
                    "rewrite".into(),
                    "write-iterative".into(),
                    "pop-culture".into(),
                ],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 9,
                name: "roleplay".into(),
                domain: "Dialogue & Roleplay".into(),
                description: "Character roleplay and dialogue".into(),
                adapter_file: "cae_09_roleplay.safetensors".into(),
                collection_tags: vec!["roleplay-characters".into(), "dialogue".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 10,
                name: "geography".into(),
                domain: "World Knowledge".into(),
                description: "Geography and world knowledge".into(),
                adapter_file: "cae_10_geography.safetensors".into(),
                collection_tags: vec!["encyclopedia-world-knowledge".into(), "geography".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 11,
                name: "history".into(),
                domain: "History".into(),
                description: "Historical knowledge and timelines".into(),
                adapter_file: "cae_11_history.safetensors".into(),
                collection_tags: vec!["history".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 12,
                name: "agent-tools".into(),
                domain: "Tool-Use Agents".into(),
                description: "Agent tools, function calling, and tool use".into(),
                adapter_file: "cae_12_agent_tools.safetensors".into(),
                collection_tags: vec![
                    "agent-tools".into(),
                    "traces".into(),
                    "function-calling".into(),
                    "holodeck".into(),
                ],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 13,
                name: "reasoning".into(),
                domain: "Deep Reasoning".into(),
                description: "Advanced reasoning from Kimi/Opus/DeepSeek traces".into(),
                adapter_file: "cae_13_reasoning.safetensors".into(),
                collection_tags: vec!["reasoning".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 14,
                name: "general-knowledge".into(),
                domain: "General Knowledge".into(),
                description: "Trivia, encyclopedia, and general facts".into(),
                adapter_file: "cae_14_general_knowledge.safetensors".into(),
                collection_tags: vec![
                    "encyclopedia".into(),
                    "wikipedia-qa".into(),
                    "trivia".into(),
                ],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 15,
                name: "psychology".into(),
                domain: "Human Behavior".into(),
                description: "Psychology, emotion, and human behavior".into(),
                adapter_file: "cae_15_psychology.safetensors".into(),
                collection_tags: vec!["emotion-psychology".into(), "human-behavior".into()],
                can_review: true,
                can_draft: true,
            },
            CaeExpert {
                id: 16,
                name: "art-music".into(),
                domain: "Creative Arts".into(),
                description: "Art, music theory, and creative expression".into(),
                adapter_file: "cae_16_art_music.safetensors".into(),
                collection_tags: vec!["art".into(), "music".into()],
                can_review: true,
                can_draft: true,
            },
        ];

        Self { experts }
    }

    pub fn get(&self, id: usize) -> Option<&CaeExpert> {
        self.experts.iter().find(|e| e.id == id)
    }

    pub fn get_by_name(&self, name: &str) -> Option<&CaeExpert> {
        self.experts.iter().find(|e| e.name == name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.experts.iter().map(|e| e.name.as_str()).collect()
    }

    pub fn count(&self) -> usize {
        self.experts.len()
    }
}

impl Default for CaeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_16_experts() {
        let registry = CaeRegistry::new();
        assert_eq!(registry.count(), 16);
    }

    #[test]
    fn registry_experts_have_unique_ids() {
        let registry = CaeRegistry::new();
        let mut ids: Vec<usize> = registry.experts.iter().map(|e| e.id).collect();
        ids.sort();
        assert_eq!(ids, (1..=16).collect::<Vec<_>>());
    }

    #[test]
    fn registry_lookup_by_id() {
        let registry = CaeRegistry::new();
        let math = registry.get(1).unwrap();
        assert_eq!(math.name, "math");
    }

    #[test]
    fn registry_lookup_by_name() {
        let registry = CaeRegistry::new();
        let math = registry.get_by_name("math").unwrap();
        assert_eq!(math.id, 1);
    }
}
