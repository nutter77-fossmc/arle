use std::process::ExitCode;

use crate::args::{CaeArgs, CaeCommand, CaeServeArgs};
use cae::config::CaeConfig;
use cae::pipeline::CaePipeline;
use cae::registry::CaeRegistry;

fn print_experts(registry: &CaeRegistry) {
    println!("CAE Experts ({} total):", registry.count());
    println!("{:-<72}", "");
    println!(
        " {:<4} {:<20} {:<18} {:<8} {:<8}",
        "ID", "Name", "Domain", "Drafts", "Reviews"
    );
    println!("{:-<72}", "");
    for expert in &registry.experts {
        println!(
            " {:<4} {:<20} {:<18} {:<8} {:<8}",
            expert.id,
            expert.name,
            expert.domain,
            if expert.can_draft { "yes" } else { "no" },
            if expert.can_review { "yes" } else { "no" },
        );
    }
}

pub(crate) fn run_cae(command: CaeArgs) -> ExitCode {
    match command.command {
        CaeCommand::List => run_cae_list(),
        CaeCommand::Status => run_cae_status(),
        CaeCommand::Serve(args) => run_cae_serve(*args),
    }
}

fn run_cae_list() -> ExitCode {
    let registry = CaeRegistry::new();
    print_experts(&registry);
    ExitCode::SUCCESS
}

fn run_cae_status() -> ExitCode {
    let registry = CaeRegistry::new();
    print_experts(&registry);
    println!();
    println!("Pipeline status: idle");
    println!("Active adapter: none");
    println!("Aggregator model: Qwen3.5-2B");
    println!("Expert base model: Qwen3.5-0.8B");
    ExitCode::SUCCESS
}

#[cfg(feature = "metal")]
fn run_cae_serve(args: CaeServeArgs) -> ExitCode {
    use crate::cae_engine::MetalCaeEngine;

    eprintln!("[CAE] Loading expert base model...");
    let config = match std::fs::read_to_string(&args.config) {
        Ok(s) => serde_json::from_str::<CaeConfig>(&s).unwrap_or_else(|e| {
            eprintln!("[CAE] Warning: failed to parse config: {e}, using defaults");
            CaeConfig::default_m1()
        }),
        Err(_) => {
            eprintln!("[CAE] No config found at {}, using defaults", args.config);
            CaeConfig::default_m1()
        }
    };

    let engine = match MetalCaeEngine::new(&config.base_expert_model_path) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("[CAE] Failed to load model: {err}");
            return ExitCode::FAILURE;
        }
    };

    let mut pipeline = CaePipeline::new(config).with_inference_provider(Box::new(engine));
    eprintln!("[CAE] Pipeline ready on port {}", args.port);

    // Interactive REPL loop
    loop {
        let mut input = String::new();
        eprint!("> ");
        if std::io::stdin().read_line(&mut input).is_err() || input.trim().is_empty() {
            continue;
        }
        let query = input.trim();
        if query == "exit" || query == "quit" {
            break;
        }

        match pipeline.execute(query) {
            Ok(result) => {
                println!("{}", result.final_response);
                eprintln!(
                    "[CAE] {} turns, {} experts, {} ms",
                    result.turns.len(),
                    result.expert_count,
                    result.total_duration_ms
                );
            }
            Err(err) => {
                eprintln!("[CAE] Pipeline error: {err}");
            }
        }
    }

    ExitCode::SUCCESS
}

#[cfg(not(feature = "metal"))]
fn run_cae_serve(_args: CaeServeArgs) -> ExitCode {
    eprintln!("[CAE] Metal backend required. Build with: --features metal,no-cuda");
    ExitCode::FAILURE
}
