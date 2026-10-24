// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
#![forbid(unsafe_code)]
pub mod dataset;
pub mod evaluation;
pub mod geometry;
pub mod model;
pub mod pose;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
