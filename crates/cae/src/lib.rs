//! # CAE — Communicative Agentic Experts Pipeline
//!
//! A Mixture of Agents (MoA) pipeline running on Apple Silicon,
//! using Qwen3.5-2B as the aggregator and 16 Qwen3.5-0.8B LoRA-fine-tuned
//! experts that collaborate via a draft-review-revise-assess-submit pipeline.
//!
//! This crate integrates into the `arle` workspace, using its Metal backend
//! for inference and its training infrastructure for LoRA adaptation.

pub mod adapter;
pub mod config;
pub mod engine;
pub mod pipeline;
pub mod registry;

pub use adapter::{AdapterInfo, AdapterManager, AdapterState};
pub use config::{CaeConfig, PipelineStep};
pub use engine::InferenceProvider;
pub use pipeline::{CaePipeline, PipelineResult, PipelineTurn, RoutingPlan};
pub use registry::{CaeExpert, CaeRegistry};
