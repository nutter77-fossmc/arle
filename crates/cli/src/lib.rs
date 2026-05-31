mod args;
mod cae;
#[cfg(feature = "metal")]
mod cae_engine;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod banner;
mod doctor;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod download;
mod hardware;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod hf_search;
mod hub_discovery;
mod model_catalog;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod model_picker;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod modelscope;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod repl;
mod serve;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod startup;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod tps;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod trace;
mod train_cli;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
mod welcome;

use std::process::ExitCode;
