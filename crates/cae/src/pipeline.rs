use crate::config::{CaeConfig, PipelineStep};
use crate::engine::InferenceProvider;
use crate::registry::CaeRegistry;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTurn {
    pub step: PipelineStep,
    pub expert_name: String,
    pub input: String,
    pub output: String,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub query: String,
    pub final_response: String,
    pub turns: Vec<PipelineTurn>,
    pub total_duration_ms: u64,
    pub expert_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingPlan {
    pub experts: Vec<String>,
    pub steps: Vec<String>,
    pub reasoning: String,
}

pub struct CaePipeline {
    config: CaeConfig,
    registry: CaeRegistry,
    context: Vec<String>,
    infer: Option<Box<dyn InferenceProvider>>,
}

impl CaePipeline {
    pub fn new(config: CaeConfig) -> Self {
        Self {
            registry: CaeRegistry::new(),
            config,
            context: Vec::new(),
            infer: None,
        }
    }

    pub fn with_inference_provider(mut self, provider: Box<dyn InferenceProvider>) -> Self {
        self.infer = Some(provider);
        self
    }

    pub fn set_inference_provider(&mut self, provider: Box<dyn InferenceProvider>) {
        self.infer = Some(provider);
    }

    pub fn registry(&self) -> &CaeRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut CaeRegistry {
        &mut self.registry
    }

    pub fn context(&self) -> &[String] {
        &self.context
    }

    pub fn add_to_context(&mut self, entry: String) {
        self.context.push(entry);
    }

    pub fn route_query(&self, query: &str) -> RoutingPlan {
        let query_lower = query.to_lowercase();
        let mut experts: Vec<String> = Vec::new();

        if query_lower.contains("math")
            || query_lower.contains("calculate")
            || query_lower.contains("equation")
            || query_lower.contains("solve")
        {
            experts.push("math".into());
        }
        if query_lower.contains("code")
            || query_lower.contains("program")
            || query_lower.contains("function")
            || query_lower.contains("debug")
        {
            experts.push("code".into());
        }
        if query_lower.contains("rust")
            || query_lower.contains("cargo")
            || query_lower.contains("unsafe")
        {
            experts.push("rust".into());
        }
        if query_lower.contains("swift") || query_lower.contains("ios") || query_lower.contains("apple") {
            experts.push("swift".into());
        }
        if query_lower.contains("physics") || query_lower.contains("force") || query_lower.contains("energy") {
            experts.push("physics".into());
        }
        if query_lower.contains("biology")
            || query_lower.contains("dna")
            || query_lower.contains("cell")
            || query_lower.contains("organism")
        {
            experts.push("biology".into());
        }
        if query_lower.contains("chemistry") || query_lower.contains("reaction") || query_lower.contains("molecule") {
            experts.push("chemistry".into());
        }
        if query_lower.contains("write")
            || query_lower.contains("story")
            || query_lower.contains("poem")
            || query_lower.contains("creative")
        {
            experts.push("creative-writing".into());
        }
        if query_lower.contains("roleplay")
            || query_lower.contains("character")
            || query_lower.contains("dialogue")
        {
            experts.push("roleplay".into());
        }
        if query_lower.contains("geography")
            || query_lower.contains("country")
            || query_lower.contains("map")
            || query_lower.contains("capital")
        {
            experts.push("geography".into());
        }
        if query_lower.contains("history") || query_lower.contains("century") || query_lower.contains("ancient") {
            experts.push("history".into());
        }
        if query_lower.contains("agent")
            || query_lower.contains("tool")
            || query_lower.contains("api")
            || query_lower.contains("function call")
        {
            experts.push("agent-tools".into());
        }
        if query_lower.contains("reason")
            || query_lower.contains("think")
            || query_lower.contains("logic")
            || query_lower.contains("deduce")
        {
            experts.push("reasoning".into());
        }
        if query_lower.contains("psychology")
            || query_lower.contains("emotion")
            || query_lower.contains("behavior")
            || query_lower.contains("mental")
        {
            experts.push("psychology".into());
        }
        if query_lower.contains("art")
            || query_lower.contains("music")
            || query_lower.contains("paint")
            || query_lower.contains("song")
        {
            experts.push("art-music".into());
        }

        if experts.is_empty() {
            experts.push("general-knowledge".into());
        }

        let mut seen = std::collections::HashSet::new();
        experts.retain(|e| seen.insert(e.clone()));

        let mut steps = vec!["draft".to_string()];
        if experts.len() > 1 {
            steps.push("review".to_string());
            steps.push("revise".to_string());
        }
        steps.push("assess".to_string());
        steps.push("submit".to_string());

        RoutingPlan {
            reasoning: format!(
                "Query matches {} expert(s): {}. Executing pipeline.",
                experts.len(),
                experts.join(", ")
            ),
            steps,
            experts,
        }
    }

    pub fn execute(&mut self, query: &str) -> Result<PipelineResult> {
        let start = std::time::Instant::now();
        let mut turns = Vec::new();

        let plan = self.route_query(query);
        let plan_turn = PipelineTurn {
            step: PipelineStep::Plan,
            expert_name: "aggregator".into(),
            input: query.to_string(),
            output: plan.reasoning.clone(),
            duration_ms: 0,
        };
        turns.push(plan_turn);

        for (i, expert_name) in plan.experts.iter().enumerate() {
            let (step, next_step) = if i == 0 {
                (PipelineStep::Draft, PipelineStep::Revise)
            } else {
                (PipelineStep::Review, PipelineStep::Revise)
            };

            let turn_start = std::time::Instant::now();
            let output = self.run_expert_step(expert_name, query, step)?;
            let turn = PipelineTurn {
                step,
                expert_name: expert_name.clone(),
                input: query.to_string(),
                output: output.clone(),
                duration_ms: turn_start.elapsed().as_millis() as u64,
            };
            turns.push(turn);

            let revise_start = std::time::Instant::now();
            let revised = self.run_expert_step(expert_name, &output, next_step)?;
            let revise_turn = PipelineTurn {
                step: next_step,
                expert_name: expert_name.clone(),
                input: output.clone(),
                output: revised.clone(),
                duration_ms: revise_start.elapsed().as_millis() as u64,
            };
            turns.push(revise_turn);
        }

        let assess_output = self.run_aggregator(query, PipelineStep::Assess)?;
        turns.push(PipelineTurn {
            step: PipelineStep::Assess,
            expert_name: "aggregator".into(),
            input: query.to_string(),
            output: assess_output.clone(),
            duration_ms: 0,
        });

        let final_response = assess_output;
        turns.push(PipelineTurn {
            step: PipelineStep::Submit,
            expert_name: "aggregator".into(),
            input: query.to_string(),
            output: final_response.clone(),
            duration_ms: 0,
        });

        self.context.push(format!("Q: {} | A: {}", query, final_response));

        let result = PipelineResult {
            query: query.to_string(),
            final_response,
            turns,
            total_duration_ms: start.elapsed().as_millis() as u64,
            expert_count: plan.experts.len(),
        };

        Ok(result)
    }

    fn run_expert_step(
        &mut self,
        expert_name: &str,
        input: &str,
        step: PipelineStep,
    ) -> Result<String> {
        let _expert = self
            .registry
            .get_by_name(expert_name)
            .context(format!("Unknown expert: {}", expert_name))?;

        if let Some(infer) = &mut self.infer {
            let sys_prompt = format!(
                "You are {} CAE-{}, a domain expert. Respond concisely.\nUser: {}",
                _step_label(step),
                expert_name,
                input
            );
            infer.generate(&sys_prompt, self.config.inference.max_tokens)
        } else {
            Ok(format!(
                "[{} CAE-{} analysis]\n{}",
                _step_label(step),
                expert_name,
                input
            ))
        }
    }

    fn run_aggregator(&mut self, input: &str, _step: PipelineStep) -> Result<String> {
        if let Some(infer) = &mut self.infer {
            let sys_prompt = format!(
                "You are the CAE aggregator. Synthesize expert responses.\nContext: {}",
                input
            );
            infer.generate(&sys_prompt, self.config.inference.max_tokens * 2)
        } else {
            Ok(format!(
                "[Aggregator assessment]\nEvaluated and synthesized: {}",
                input
            ))
        }
    }
}

fn _step_label(step: PipelineStep) -> &'static str {
    match step {
        PipelineStep::Plan => "",
        PipelineStep::Draft => "Draft",
        PipelineStep::Review => "Review",
        PipelineStep::Revise => "Revise",
        PipelineStep::Assess => "Assess",
        PipelineStep::Submit => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_query_math() {
        let pipeline = CaePipeline::new(CaeConfig::default_m1());
        let plan = pipeline.route_query("solve this equation: 2x + 3 = 7");
        assert!(plan.experts.contains(&"math".to_string()));
    }

    #[test]
    fn route_query_code() {
        let pipeline = CaePipeline::new(CaeConfig::default_m1());
        let plan = pipeline.route_query("write a function to sort an array");
        assert!(plan.experts.contains(&"code".to_string()));
    }

    #[test]
    fn route_query_multi_expert() {
        let pipeline = CaePipeline::new(CaeConfig::default_m1());
        let plan = pipeline.route_query("write a rust program to calculate fibonacci");
        assert!(plan.experts.contains(&"rust".to_string()));
        assert!(plan.experts.contains(&"code".to_string()));
        assert!(plan.experts.contains(&"math".to_string()));
    }

    #[test]
    fn route_query_fallback_to_general() {
        let pipeline = CaePipeline::new(CaeConfig::default_m1());
        let plan = pipeline.route_query("what is the weather today?");
        assert!(plan.experts.contains(&"general-knowledge".to_string()));
    }

    #[test]
    fn pipeline_executes_without_error() {
        let mut pipeline = CaePipeline::new(CaeConfig::default_m1());
        let result = pipeline.execute("what is 2+2?").unwrap();
        assert_eq!(result.query, "what is 2+2?");
        assert!(!result.turns.is_empty());
    }
}
